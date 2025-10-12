#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "lingodb/mlir-support/parsing.h"
#include "mlir/Support/LLVM.h"

#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "runtime-defs/Hashtable.h"
#include "runtime-defs/LazyJoinHashtable.h"
#include "runtime-defs/PgSortRuntime.h"
#include "runtime-defs/TableBuilder.h"
#include "runtime-defs/Vector.h"
#include "lingodb/runtime/RuntimeSpecifications.h"
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
#include "utils/memutils.h"
}

using namespace mlir;
namespace {
static mlir::util::RefType getLoweredVectorType(MLIRContext* context, Type elementType) {
   auto sType = mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), mlir::util::RefType::get(context, elementType)});
   return mlir::util::RefType::get(context, sType);
}
static Type getHashtableKVType(MLIRContext* context, Type keyType, Type aggrType) {
   return mlir::TupleType::get(context, {keyType, aggrType});
}
static TupleType getHashtableEntryType(MLIRContext* context, Type keyType, Type aggrType) {
   auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
   return mlir::TupleType::get(context, {i8PtrType, IndexType::get(context), getHashtableKVType(context, keyType, aggrType)});
}
static runtime::HashtableSpecification* createHashtableSpecFromTypes(mlir::Type keyType, mlir::Type valType) {
    auto extractColumns = [](mlir::Type tupleType) -> std::vector<std::pair<uint32_t, bool>> {
        std::vector<std::pair<uint32_t, bool>> columns;
        if (auto tuple = tupleType.dyn_cast<mlir::TupleType>()) {
            for (auto fieldType : tuple.getTypes()) {
                bool nullable = false;
                mlir::Type baseType = fieldType;

                if (auto nullableType = fieldType.dyn_cast<mlir::db::NullableType>()) {
                    nullable = true;
                    baseType = nullableType.getType();
                }

                // Map MLIR type to PostgreSQL OID
                uint32_t oid = INT4OID; // default
                if (baseType.isInteger(32)) {
                    oid = INT4OID;
                } else if (baseType.isInteger(64)) {
                    oid = INT8OID;
                } else if (auto stringType = baseType.dyn_cast<mlir::db::StringType>()) {
                    oid = TEXTOID;
                } else if (auto decimalType = baseType.dyn_cast<mlir::db::DecimalType>()) {
                    oid = NUMERICOID;
                }

                columns.push_back({oid, nullable});
            }
        }
        return columns;
    };

    auto keyColumns = extractColumns(keyType);
    auto valColumns = extractColumns(valType);

    if (keyColumns.empty() && valColumns.empty()) {
        return nullptr;
    }

    MemoryContext oldContext = MemoryContextSwitchTo(CurTransactionContext);

    auto* spec = static_cast<runtime::HashtableSpecification*>(
        MemoryContextAlloc(CurTransactionContext, sizeof(runtime::HashtableSpecification)));

    spec->num_key_columns = keyColumns.size();
    spec->num_value_columns = valColumns.size();

    spec->key_columns = static_cast<runtime::HashtableColumnInfo*>(
        MemoryContextAlloc(CurTransactionContext, keyColumns.size() * sizeof(runtime::HashtableColumnInfo)));

    spec->value_columns = static_cast<runtime::HashtableColumnInfo*>(
        MemoryContextAlloc(CurTransactionContext, valColumns.size() * sizeof(runtime::HashtableColumnInfo)));

    for (size_t i = 0; i < keyColumns.size(); i++) {
        spec->key_columns[i].type_oid = keyColumns[i].first;
        spec->key_columns[i].is_nullable = keyColumns[i].second;
    }

    for (size_t i = 0; i < valColumns.size(); i++) {
        spec->value_columns[i].type_oid = valColumns[i].first;
        spec->value_columns[i].is_nullable = valColumns[i].second;
    }

    MemoryContextSwitchTo(oldContext);

    return spec;
}

static Type getHashtableType(MLIRContext* context, Type keyType, Type aggrType) {
   auto idxType = IndexType::get(context);
   auto entryType = getHashtableEntryType(context, keyType, aggrType);
   auto entryPtrType = mlir::util::RefType::get(context, entryType);
   auto valuesType = mlir::util::RefType::get(context, entryType);
   auto htType = mlir::util::RefType::get(context, entryPtrType);
   auto tplType = mlir::TupleType::get(context, {htType, idxType, idxType, valuesType, idxType, aggrType});
   return mlir::util::RefType::get(context, tplType);
}

class CreateDsLowering : public OpConversionPattern<mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = createOp->getLoc();
      if (auto genericType = createOp.getDs().getType().dyn_cast<mlir::dsa::GenericIterableType>()) {
         if (genericType.getIteratorName() == "pgsort_iterator") {
            // Create PgSortState - extract spec pointer from attribute
            auto elementType = typeConverter->convertType(genericType.getElementType());
            auto typeSize = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);

            // Extract SortSpecification pointer from InitialCapacity attribute
            mlir::Value specPtrValue;
            if (auto specAttr = createOp.getInitAttr()) {
               // specAttr is an IntegerAttr with the spec pointer value (already signless)
               if (auto intAttr = specAttr->dyn_cast<mlir::IntegerAttr>()) {
                  specPtrValue = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
               } else {
                  // Unexpected attribute type - pass 0
                  specPtrValue = rewriter.create<mlir::arith::ConstantOp>(
                     loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0)
                  );
               }
            } else {
               // No spec provided - pass 0
               specPtrValue = rewriter.create<mlir::arith::ConstantOp>(
                  loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0)
               );
            }

            auto ptr = rt::PgSortState::create(rewriter, loc)({typeSize, specPtrValue})[0];
            rewriter.replaceOp(createOp, ptr);
            return success();
         }
      } else if (auto joinHtType = createOp.getDs().getType().dyn_cast<mlir::dsa::JoinHashtableType>()) {
         auto entryType = mlir::TupleType::get(rewriter.getContext(), {joinHtType.getKeyType(), joinHtType.getValType()});
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
         Value typesize = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), typeConverter->convertType(tupleType));

         Value specPtr;
         auto* spec = createHashtableSpecFromTypes(joinHtType.getKeyType(), joinHtType.getValType());
         if (spec) {
            specPtr = rewriter.create<arith::ConstantIndexOp>(loc, reinterpret_cast<uint64_t>(spec));
         } else {
            specPtr = rewriter.create<arith::ConstantIndexOp>(loc, 0);
         }

         Value ptr = rt::LazyJoinHashtable::create(rewriter, loc)({typesize, specPtr})[0];
         rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(createOp, typeConverter->convertType(joinHtType), ptr);
         return success();
      } else if (auto vecType = createOp.getDs().getType().dyn_cast<mlir::dsa::VectorType>()) {
         auto elementType = typeConverter->convertType(vecType.getElementType());
         auto typeSize = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);

         if (auto initAttr = createOp.getInitAttr()) {
            if (auto specPtrAttr = initAttr->dyn_cast<mlir::IntegerAttr>()) {
               auto ptr = rt::PgSortState::create(rewriter, loc)({typeSize})[0];
               mlir::Value createdVector = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, getLoweredVectorType(rewriter.getContext(), elementType), ptr);
               rewriter.replaceOp(createOp, createdVector);
               return success();
            }
         }

         Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, 1024);
         auto ptr = rt::Vector::create(rewriter, loc)({typeSize, initialCapacity})[0];
         mlir::Value createdVector = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, getLoweredVectorType(rewriter.getContext(), elementType), ptr);
         rewriter.replaceOp(createOp, createdVector);
         return success();
      } else if (auto aggrHtType = createOp.getDs().getType().dyn_cast<mlir::dsa::AggregationHashtableType>()) {
         TupleType keyType = aggrHtType.getKeyType();
         TupleType aggrType = aggrHtType.getValType();
         if (keyType.getTypes().empty()) {
            ::mlir::Value ref = rewriter.create<mlir::util::AllocOp>(loc, typeConverter->convertType(createOp.getDs().getType()), mlir::Value());
            rewriter.create<mlir::util::StoreOp>(loc, adaptor.getInitVal(), ref, ::mlir::Value());
            rewriter.replaceOp(createOp, ref);
            return success();
         } else {
            auto typeSize = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), getHashtableEntryType(rewriter.getContext(), keyType, aggrType));
            Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, 4);

            Value specPtrValue;
            if (auto specAttr = createOp->getAttrOfType<mlir::IntegerAttr>("spec_ptr")) {
               uint64_t ptrVal = specAttr.getValue().getZExtValue();
               // Create ConstantOp with I64 type (not ConstantIndexOp!)
               specPtrValue = rewriter.create<mlir::arith::ConstantOp>(
                  loc, rewriter.getIntegerAttr(rewriter.getI64Type(), ptrVal)
               );
            } else {
               // PGX-LOWER: Create spec from AggregationHashtableType's key/value types
               auto* spec = createHashtableSpecFromTypes(keyType, aggrType);
               if (spec) {
                  specPtrValue = rewriter.create<mlir::arith::ConstantOp>(
                     loc, rewriter.getIntegerAttr(rewriter.getI64Type(), reinterpret_cast<uint64_t>(spec))
                  );
               } else {
                  specPtrValue = rewriter.create<mlir::arith::ConstantOp>(
                     loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0)
                  );
               }
            }

            auto ptr = rt::Hashtable::create(rewriter, loc)({typeSize, initialCapacity, specPtrValue})[0];
            mlir::Value casted = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, getHashtableType(rewriter.getContext(), keyType, aggrType), ptr);
            Value initValAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), adaptor.getInitVal().getType()), casted, 5);
            rewriter.create<mlir::util::StoreOp>(loc, adaptor.getInitVal(), initValAddress, Value());
            rewriter.replaceOp(createOp, casted);
            return success();
         }
      }
      return failure();
   }
};
class HtInsertLowering : public OpConversionPattern<mlir::dsa::HashtableInsert> {
   public:
   using OpConversionPattern<mlir::dsa::HashtableInsert>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::HashtableInsert insertOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!insertOp.getHt().getType().isa<mlir::dsa::AggregationHashtableType>()) {
         return failure();
      }
      std::function<Value(OpBuilder&, Value, Value)> reduceFnBuilder = insertOp.getReduce().empty() ? std::function<Value(OpBuilder&, Value, Value)>() : [&insertOp](OpBuilder& rewriter, Value left, Value right) {
         Block* sortLambda = &insertOp.getReduce().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::IRMapping mapping;
         mapping.map(sortLambda->getArgument(0), left);
         mapping.map(sortLambda->getArgument(1), right);

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         return mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).getResults()[0]);
      };
      std::function<Value(OpBuilder&, Value, Value)> equalFnBuilder = insertOp.getEqual().empty() ? std::function<Value(OpBuilder&, Value, Value)>() : [&insertOp](OpBuilder& rewriter, Value left, Value right) {
         Block* sortLambda = &insertOp.getEqual().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::IRMapping mapping;
         mapping.map(sortLambda->getArgument(0), left);
         mapping.map(sortLambda->getArgument(1), right);

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         return mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).getResults()[0]);
      };
      auto loc = insertOp->getLoc();
      if (insertOp.getHt().getType().cast<mlir::dsa::AggregationHashtableType>().getKeyType() == mlir::TupleType::get(getContext())) {
         auto loaded = rewriter.create<mlir::util::LoadOp>(loc, adaptor.getHt().getType().cast<mlir::util::RefType>().getElementType(), adaptor.getHt(), mlir::Value());
         auto newAggr = reduceFnBuilder(rewriter, loaded, adaptor.getVal());
         rewriter.create<mlir::util::StoreOp>(loc, newAggr, adaptor.getHt(), mlir::Value());
         rewriter.eraseOp(insertOp);
      } else {
         Value hashed;
         {
            Block* sortLambda = &insertOp.getHash().front();
            auto* sortLambdaTerminator = sortLambda->getTerminator();
            mlir::IRMapping mapping;
            mapping.map(sortLambda->getArgument(0), adaptor.getKey());

            for (auto& op : sortLambda->getOperations()) {
               if (&op != sortLambdaTerminator) {
                  rewriter.clone(op, mapping);
               }
            }
            hashed = mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).getResults()[0]);
         }

         auto keyType = adaptor.getKey().getType();
         auto aggrType = typeConverter->convertType(insertOp.getHt().getType().cast<mlir::dsa::AggregationHashtableType>().getValType());
         auto* context = rewriter.getContext();
         auto entryType = getHashtableEntryType(context, keyType, aggrType);
         auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
         auto idxType = rewriter.getIndexType();
         auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
         auto kvType = getHashtableKVType(context, keyType, aggrType);
         auto kvPtrType = mlir::util::RefType::get(context, kvType);
         auto keyPtrType = mlir::util::RefType::get(context, keyType);
         auto aggrPtrType = mlir::util::RefType::get(context, aggrType);
         auto valuesType = mlir::util::RefType::get(context, entryType);
         auto entryPtrType = mlir::util::RefType::get(context, entryType);
         auto htType = mlir::util::RefType::get(context, entryPtrType);

         Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.getHt(), 1);
         Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.getHt(), 2);

         Value len = rewriter.create<util::LoadOp>(loc, idxType, lenAddress);
         Value capacityInitial = rewriter.create<util::LoadOp>(loc, idxType, capacityAddress);

         Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
         Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);

         Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacityInitial);
         rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
               rt::Hashtable::resize(b,loc)(adaptor.getHt());
               b.create<scf::YieldOp>(loc); });

         Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), htType), adaptor.getHt(), 0);
         Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);

         Value capacity = rewriter.create<util::LoadOp>(loc, idxType, capacityAddress);

         Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         Value htSize = rewriter.create<arith::MulIOp>(loc, capacity, two);

         Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);
         Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
         Type bucketPtrType = util::RefType::get(context, entryType);
         Type ptrType = util::RefType::get(context, bucketPtrType);
         Type doneType = rewriter.getI1Type();
         Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, ptrType, ht, position);

         auto resultTypes = std::vector<Type>({ptrType});
         auto whileOp = rewriter.create<scf::WhileOp>(loc, resultTypes, ValueRange({ptr}));
         Block* before = rewriter.createBlock(&whileOp.getBefore(), {}, resultTypes, {loc});
         Block* after = rewriter.createBlock(&whileOp.getAfter(), {}, resultTypes, {loc});

         {
            rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
            Value ptr = before->getArgument(0);

            Value currEntryPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ptr);
            Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
            auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange({doneType, ptrType}), cmp);
            {
               OpBuilder::InsertionGuard guard(rewriter);
               rewriter.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
               OpBuilder& b = rewriter;

                  Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
                  Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                  auto ifOpH = b.create<scf::IfOp>(loc, TypeRange({doneType,ptrType}), hashMatches);
                  {
                     OpBuilder::InsertionGuard guard(b);
                     b.setInsertionPointToStart(&ifOpH.getThenRegion().emplaceBlock());
                        Value kvAddress=rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                        Value entryKeyAddress=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                        Value entryKey = rewriter.create<util::LoadOp>(loc, keyType, entryKeyAddress);

                        Value keyMatches = equalFnBuilder(b,entryKey,adaptor.getKey());
                        auto ifOp2 = b.create<scf::IfOp>(loc, TypeRange({doneType,ptrType}), keyMatches);
                        {
                           OpBuilder::InsertionGuard guard2(b);
                           b.setInsertionPointToStart(&ifOp2.getThenRegion().emplaceBlock());
                              if(reduceFnBuilder) {
                                 Value entryAggrAddress = rewriter.create<util::TupleElementPtrOp>(loc, aggrPtrType, kvAddress, 1);
                                 Value entryAggr = rewriter.create<util::LoadOp>(loc, aggrType, entryAggrAddress);
                                 Value newAggr = reduceFnBuilder(b, entryAggr, adaptor.getVal());
                                 b.create<util::StoreOp>(loc, newAggr, entryAggrAddress, Value());
                              }
                              b.create<scf::YieldOp>(loc, ValueRange{falseValue,ptr});
                              
                              // Else branch of ifOp2
                              b.setInsertionPointToStart(&ifOp2.getElseRegion().emplaceBlock());
                              //          ptr = &entry.next
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, ptrType,currEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });
                        }
                        b.setInsertionPointAfter(ifOp2);
                        b.create<scf::YieldOp>(loc, ifOp2.getResults());
                        
                        // Else branch of ifOpH
                        b.setInsertionPointToStart(&ifOpH.getElseRegion().emplaceBlock());
                        //          ptr = &entry.next
                        Value newPtr=b.create<util::GenericMemrefCastOp>(loc, ptrType,currEntryPtr);
                        //          yield ptr,done=false
                        b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });
                  }
                  b.setInsertionPointAfter(ifOpH);
                  b.create<scf::YieldOp>(loc, ifOpH.getResults()); 
               
               // Else branch of outer ifOp - create new entry with deep copy
               rewriter.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
               b = rewriter;  // Keep alias for minimal changes
                  Value initValAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), aggrType), adaptor.getHt(), 5);
                  Value initialVal = b.create<util::LoadOp>(loc, aggrType, initValAddress);
                  Value newAggr = reduceFnBuilder ? reduceFnBuilder(b,initialVal, adaptor.getVal()): initialVal;

                  Value hashedI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), hashed);
                  Value lenI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), len);

                  auto keyRefType = mlir::util::RefType::get(b.getContext(), adaptor.getKey().getType());
                  auto keyRef = b.create<mlir::util::AllocaOp>(loc, keyRefType, mlir::Value());
                  b.create<mlir::util::StoreOp>(loc, adaptor.getKey(), keyRef, mlir::Value());
                  auto i8RefType = mlir::util::RefType::get(b.getContext(), b.getI8Type());
                  Value keyPtr = b.create<mlir::util::GenericMemrefCastOp>(loc, i8RefType, keyRef);

                  auto valueRefType = mlir::util::RefType::get(b.getContext(), newAggr.getType());
                  auto valueRef = b.create<mlir::util::AllocaOp>(loc, valueRefType, mlir::Value());
                  b.create<mlir::util::StoreOp>(loc, newAggr, valueRef, mlir::Value());
                  Value valuePtr = b.create<mlir::util::GenericMemrefCastOp>(loc, i8RefType, valueRef);

                  Value newValueLocPtr = rt::Hashtable::appendEntryWithDeepCopy(b, loc)(
                     {adaptor.getHt(), hashedI64, lenI64, keyPtr, valuePtr}
                  )[0];

                  newValueLocPtr = b.create<util::GenericMemrefCastOp>(loc, bucketPtrType, newValueLocPtr);

                  b.create<util::StoreOp>(loc, newValueLocPtr, ptr, Value());
                  Value newLen = b.create<arith::AddIOp>(loc, len, one);
                  b.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());

                  b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); 
            }

            Value done = ifOp.getResult(0);
            Value newPtr = ifOp.getResult(1);
            rewriter.create<scf::ConditionOp>(loc, done,
                                              ValueRange({newPtr}));
         }

         {
            rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
            rewriter.create<scf::YieldOp>(loc, after->getArguments());
         }

         rewriter.setInsertionPointAfter(whileOp);

         rewriter.eraseOp(insertOp);
      }
      return success();
   }
};
class LazyJHtInsertLowering : public OpConversionPattern<mlir::dsa::HashtableInsert> {
   public:
   using OpConversionPattern<mlir::dsa::HashtableInsert>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::HashtableInsert insertOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!insertOp.getHt().getType().isa<mlir::dsa::JoinHashtableType>()) {
         return failure();
      }
      Value hashed;
      {
         Block* sortLambda = &insertOp.getHash().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         ::mlir::IRMapping mapping;
         mapping.map(sortLambda->getArgument(0), adaptor.getKey());

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         hashed = mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).getResults()[0]);
      }
      mlir::Value val = adaptor.getVal();
      auto loc = insertOp->getLoc();

      if (!val) {
         val = rewriter.create<mlir::util::UndefOp>(loc, mlir::TupleType::get(getContext()));
      }
      auto entry = rewriter.create<mlir::util::PackOp>(loc, mlir::ValueRange({adaptor.getKey(), val}));
      auto bucket = rewriter.create<mlir::util::PackOp>(loc, mlir::ValueRange({hashed, entry}));
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto valuesType = mlir::util::RefType::get(rewriter.getContext(), bucket.getType());
      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.getHt(), 2);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.getHt(), 3);

      auto len = rewriter.create<mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = rewriter.create<mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      rewriter.create<scf::IfOp>(
         loc, cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            rt::LazyJoinHashtable::resize(b,loc)(adaptor.getHt());
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(getContext(), adaptor.getHt().getType().cast<mlir::util::RefType>().getElementType().cast<mlir::TupleType>().getType(4)), adaptor.getHt(), 4);
      Value castedValuesAddress = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(getContext(), valuesType), valuesAddress);
      auto values = rewriter.create<mlir::util::LoadOp>(loc, valuesType, castedValuesAddress, Value());
      rewriter.create<util::StoreOp>(loc, bucket, values, len);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = rewriter.create<arith::AddIOp>(loc, len, one);

      rewriter.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
      rewriter.eraseOp(insertOp);
      return success();
   }
};
class FinalizeLowering : public OpConversionPattern<mlir::dsa::Finalize> {
   public:
   using OpConversionPattern<mlir::dsa::Finalize>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Finalize finalizeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!finalizeOp.getHt().getType().isa<mlir::dsa::JoinHashtableType>()) {
         return failure();
      }
      rt::LazyJoinHashtable::finalize(rewriter, finalizeOp->getLoc())(adaptor.getHt());
      rewriter.eraseOp(finalizeOp);
      return success();
   }
};
class FinalizeTBLowering : public OpConversionPattern<mlir::dsa::Finalize> {
   public:
   using OpConversionPattern<mlir::dsa::Finalize>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Finalize finalizeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!finalizeOp.getHt().getType().isa<mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      mlir::Value res = rt::TableBuilder::build(rewriter, finalizeOp->getLoc())(adaptor.getHt())[0];
      rewriter.replaceOp(finalizeOp, res);
      return success();
   }
};
class PgSortAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   public:
   using OpConversionPattern<mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      PGX_IO(DSA_LOWER);
      PGX_LOG(DSA_LOWER, DEBUG, "PgSortAppendLowering: checking dsa.ds_append");

      // Log the entire operation
      std::string opStr;
      llvm::raw_string_ostream opStream(opStr);
      appendOp->print(opStream);
      opStream.flush();
      PGX_LOG(DSA_LOWER, DEBUG, "  Full operation: %s", opStr.c_str());

      auto origType = appendOp.getDs().getType();

      auto genericType = mlir::dyn_cast_or_null<mlir::dsa::GenericIterableType>(origType);
      if (!genericType) {
         PGX_LOG(DSA_LOWER, DEBUG, "  Not GenericIterableType, skipping");
         return failure();
      }

      if (genericType.getIteratorName() != "pgsort_iterator") {
         PGX_LOG(DSA_LOWER, DEBUG, "  Not pgsort_iterator, skipping");
         return failure();
      }

      PGX_LOG(DSA_LOWER, DEBUG, "  MATCH! Lowering PgSort append");

      auto loc = appendOp->getLoc();
      Value pgSortState = adaptor.getDs();  // This is !util.ref<i8> (opaque pointer)
      Value tuple = adaptor.getVal();       // This is the packed tuple

      PGX_LOG(DSA_LOWER, DEBUG, "    Lowering PgSort append with state and tuple");

      auto tupleRefType = mlir::util::RefType::get(rewriter.getContext(), tuple.getType());
      auto tupleRef = rewriter.create<mlir::util::AllocaOp>(loc, tupleRefType, mlir::Value());

      rewriter.create<mlir::util::StoreOp>(loc, tuple, tupleRef, mlir::Value());

      auto i8RefType = mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
      auto tuplePtr = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, i8RefType, tupleRef);

      rt::PgSortState::appendTuple(rewriter, loc)({pgSortState, tuplePtr});

      rewriter.eraseOp(appendOp);

      PGX_LOG(DSA_LOWER, DEBUG, "  PgSort append lowered successfully");
      return success();
   }
};

class DSAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   public:
   using OpConversionPattern<mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      PGX_IO(DSA_LOWER);
      PGX_LOG(DSA_LOWER, DEBUG, "DSAppendLowering: checking dsa.ds_append");
      if (!appendOp.getDs().getType().isa<mlir::dsa::VectorType>()) {
         PGX_LOG(DSA_LOWER, DEBUG, "  Not a VectorType, skipping");
         return failure();
      }
      PGX_LOG(DSA_LOWER, DEBUG, "  VectorType match, lowering");
      Value builderVal = adaptor.getDs();
      Value v = adaptor.getVal();
      auto convertedElementType = typeConverter->convertType(appendOp.getDs().getType().cast<mlir::dsa::VectorType>().getElementType());
      auto loc = appendOp->getLoc();
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto valuesType = mlir::util::RefType::get(rewriter.getContext(), convertedElementType);
      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, builderVal, 0);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, builderVal, 1);

      auto len = rewriter.create<mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = rewriter.create<mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      rewriter.create<scf::IfOp>(
         loc, cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            rt::Vector::resize(b,loc)({builderVal});
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), valuesType), builderVal, 2);
      auto values = rewriter.create<mlir::util::LoadOp>(loc, valuesType, valuesAddress, Value());

      rewriter.create<util::StoreOp>(loc, v, values, len);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = rewriter.create<arith::AddIOp>(loc, len, one);
      rewriter.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
      rewriter.eraseOp(appendOp);
      return success();
   }
};

class TBAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   public:
   using OpConversionPattern<mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!appendOp.getDs().getType().isa<mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      Value builderVal = adaptor.getDs();
      Value val = adaptor.getVal();
      Value isValid = adaptor.getValid();
      auto loc = appendOp->getLoc();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      mlir::Type type = getBaseType(val.getType());
      if (isIntegerType(type, 1)) {
         rt::TableBuilder::addBool(rewriter, loc)({builderVal, isValid, val});
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         switch (intWidth) {
            case 8: rt::TableBuilder::addInt8(rewriter, loc)({builderVal, isValid, val}); break;
            case 16: rt::TableBuilder::addInt16(rewriter, loc)({builderVal, isValid, val}); break;
            case 32: rt::TableBuilder::addInt32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::TableBuilder::addInt64(rewriter, loc)({builderVal, isValid, val}); break;
            case 128:
             rt::TableBuilder::addDecimal(rewriter, loc)({builderVal, isValid, val}); break;
            default: {
               val=rewriter.create<arith::ExtUIOp>(loc,rewriter.getI64Type(),val);
               rt::TableBuilder::addFixedSized(rewriter, loc)({builderVal, isValid, val});
               break;
            }

         }
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: rt::TableBuilder::addFloat32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::TableBuilder::addFloat64(rewriter, loc)({builderVal, isValid, val}); break;
         }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::util::VarLen32Type>()) {
         rt::TableBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      }
      rewriter.eraseOp(appendOp);
      return success();
   }
};
class NextRowLowering : public OpConversionPattern<mlir::dsa::NextRow> {
   public:
   using OpConversionPattern<mlir::dsa::NextRow>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::NextRow op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::TableBuilder::nextRow(rewriter, op->getLoc())({adaptor.getBuilder()});
      rewriter.eraseOp(op);
      return success();
   }
};
class CreateTableBuilderLowering : public OpConversionPattern<mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getDs().getType().isa<mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value schema = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(getContext()), createOp.getInitAttr().value().cast<StringAttr>().str());
      Value tableBuilder = rt::TableBuilder::create(rewriter, loc)({schema})[0];
      rewriter.replaceOp(createOp, tableBuilder);
      return success();
   }
};
class FreeLowering : public OpConversionPattern<mlir::dsa::FreeOp> {
   public:
   using OpConversionPattern<mlir::dsa::FreeOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::FreeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto genericType = op.getVal().getType().dyn_cast<mlir::dsa::GenericIterableType>()) {
         if (genericType.getIteratorName() == "pgsort_iterator") {
            // TODO: Fix runtime function registration for PgSortState::destroy
            // rt::PgSortState::destroy(rewriter, op->getLoc())(ValueRange{adaptor.getVal()});
         }
      }
      if (auto aggrHtType = op.getVal().getType().dyn_cast<mlir::dsa::AggregationHashtableType>()) {
         if (aggrHtType.getKeyType().getTypes().empty()) {
         } else {
            rt::Hashtable::destroy(rewriter, op->getLoc())(ValueRange{adaptor.getVal()});
         }
      }
      if (op.getVal().getType().isa<mlir::dsa::JoinHashtableType>()) {
         rt::LazyJoinHashtable::destroy(rewriter, op->getLoc())(ValueRange{adaptor.getVal()});
      }
      if (op.getVal().getType().isa<mlir::dsa::VectorType>()) {
         rt::Vector::destroy(rewriter, op->getLoc())(ValueRange{adaptor.getVal()});
      }
      rewriter.eraseOp(op);
      return success();
   }
};

class SetDecimalScaleLowering : public OpConversionPattern<mlir::dsa::SetDecimalScaleOp> {
   public:
   using OpConversionPattern<mlir::dsa::SetDecimalScaleOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::SetDecimalScaleOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      rt::TableBuilder::setNextDecimalScale(rewriter, loc)({adaptor.getBuilder(), adaptor.getScale()});
      rewriter.eraseOp(op);
      return success();
   }
};

} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateDsLowering, HtInsertLowering, FinalizeLowering, PgSortAppendLowering, DSAppendLowering, LazyJHtInsertLowering, FreeLowering>(typeConverter, patterns.getContext());
   patterns.insert<CreateTableBuilderLowering, TBAppendLowering, FinalizeTBLowering, NextRowLowering, SetDecimalScaleLowering>(typeConverter, patterns.getContext());
   typeConverter.addConversion([&typeConverter](mlir::dsa::VectorType vectorType) {
      return getLoweredVectorType(vectorType.getContext(), typeConverter.convertType(vectorType.getElementType()));
   });
   typeConverter.addConversion([&typeConverter](mlir::dsa::AggregationHashtableType aggregationHashtableType) {
      if (aggregationHashtableType.getKeyType().getTypes().empty()) {
         return (Type) mlir::util::RefType::get(aggregationHashtableType.getContext(), typeConverter.convertType(aggregationHashtableType.getValType()));
      } else {
         return (Type) getHashtableType(aggregationHashtableType.getContext(), typeConverter.convertType(aggregationHashtableType.getKeyType()), typeConverter.convertType(aggregationHashtableType.getValType()));
      }
   });
}
} // end namespace mlir::dsa
