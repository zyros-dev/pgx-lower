#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "runtime-defs/Hashtable.h"
#include "runtime-defs/LazyJoinHashtable.h"
#include "runtime-defs/TableBuilder.h"
#include "runtime-defs/Vector.h"
using namespace mlir;
namespace {
static pgx::mlir::util::RefType getLoweredVectorType(MLIRContext* context, Type elementType) {
   auto sType = mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), pgx::mlir::util::RefType::get(context, elementType)});
   return pgx::mlir::util::RefType::get(context, sType);
}
static Type getHashtableKVType(MLIRContext* context, Type keyType, Type aggrType) {
   return mlir::TupleType::get(context, {keyType, aggrType});
}
static TupleType getHashtableEntryType(MLIRContext* context, Type keyType, Type aggrType) {
   auto i8PtrType = pgx::mlir::util::RefType::get(context, IntegerType::get(context, 8));
   return mlir::TupleType::get(context, {i8PtrType, IndexType::get(context), getHashtableKVType(context, keyType, aggrType)});
}
static Type getHashtableType(MLIRContext* context, Type keyType, Type aggrType) {
   auto idxType = IndexType::get(context);
   auto entryType = getHashtableEntryType(context, keyType, aggrType);
   auto entryPtrType = pgx::mlir::util::RefType::get(context, entryType);
   auto valuesType = pgx::mlir::util::RefType::get(context, entryType);
   auto htType = pgx::mlir::util::RefType::get(context, entryPtrType);
   auto tplType = mlir::TupleType::get(context, {htType, idxType, idxType, valuesType, idxType, aggrType});
   return pgx::mlir::util::RefType::get(context, tplType);
}

class CreateDsLowering : public OpConversionPattern<pgx::mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = createOp->getLoc();
      if (auto joinHtType = createOp.ds().getType().dyn_cast<pgx::mlir::dsa::JoinHashtableType>()) {
         auto entryType = mlir::TupleType::get(rewriter.getContext(), {joinHtType.getKeyType(), joinHtType.getValType()});
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
         Value typesize = rewriter.create<pgx::mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), typeConverter->convertType(tupleType));
         Value ptr = rt::LazyJoinHashtable::create(rewriter, loc)(typesize)[0];
         rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(createOp, typeConverter->convertType(joinHtType), ptr);
         return success();
      } else if (auto vecType = createOp.ds().getType().dyn_cast<pgx::mlir::dsa::VectorType>()) {
         Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, 1024);
         auto elementType = typeConverter->convertType(vecType.getElementType());
         auto typeSize = rewriter.create<pgx::mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
         auto ptr = rt::Vector::create(rewriter, loc)({typeSize, initialCapacity})[0];
         mlir::Value createdVector = rewriter.create<pgx::mlir::util::GenericMemrefCastOp>(loc, getLoweredVectorType(rewriter.getContext(), elementType), ptr);
         rewriter.replaceOp(createOp, createdVector);
         return success();
      } else if (auto aggrHtType = createOp.ds().getType().dyn_cast<pgx::mlir::dsa::AggregationHashtableType>()) {
         TupleType keyType = aggrHtType.getKeyType();
         TupleType aggrType = aggrHtType.getValType();
         if (keyType.getTypes().empty()) {
            mlir::Value ref = rewriter.create<pgx::mlir::util::AllocOp>(loc, typeConverter->convertType(createOp.ds().getType()), mlir::Value());
            rewriter.create<pgx::mlir::util::StoreOp>(loc, adaptor.init_val(), ref, mlir::Value());
            rewriter.replaceOp(createOp, ref);
            return success();
         } else {
            auto typeSize = rewriter.create<pgx::mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), getHashtableEntryType(rewriter.getContext(), keyType, aggrType));
            Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, 4);
            auto ptr = rt::Hashtable::create(rewriter, loc)({typeSize, initialCapacity})[0];
            mlir::Value casted = rewriter.create<pgx::mlir::util::GenericMemrefCastOp>(loc, getHashtableType(rewriter.getContext(), keyType, aggrType), ptr);
            Value initValAddress = rewriter.create<util::TupleElementPtrOp>(loc, pgx::mlir::util::RefType::get(rewriter.getContext(), adaptor.init_val().getType()), casted, 5);
            rewriter.create<pgx::mlir::util::StoreOp>(loc, adaptor.init_val(), initValAddress, Value());
            rewriter.replaceOp(createOp, casted);
            return success();
         }
      }
      return failure();
   }
};
class HtInsertLowering : public OpConversionPattern<pgx::mlir::dsa::HashtableInsert> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::HashtableInsert>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::HashtableInsert insertOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!insertOp.ht().getType().isa<pgx::mlir::dsa::AggregationHashtableType>()) {
         return failure();
      }
      std::function<Value(OpBuilder&, Value, Value)> reduceFnBuilder = insertOp.reduce().empty() ? std::function<Value(OpBuilder&, Value, Value)>() : [&insertOp](OpBuilder& rewriter, Value left, Value right) {
         Block* sortLambda = &insertOp.reduce().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::IRMapping mapping;
         mapping.map(sortLambda->getArgument(0), left);
         mapping.map(sortLambda->getArgument(1), right);

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         return mapping.lookup(cast<pgx::mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
      };
      std::function<Value(OpBuilder&, Value, Value)> equalFnBuilder = insertOp.equal().empty() ? std::function<Value(OpBuilder&, Value, Value)>() : [&insertOp](OpBuilder& rewriter, Value left, Value right) {
         Block* sortLambda = &insertOp.equal().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::IRMapping mapping;
         mapping.map(sortLambda->getArgument(0), left);
         mapping.map(sortLambda->getArgument(1), right);

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         return mapping.lookup(cast<pgx::mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
      };
      auto loc = insertOp->getLoc();
      if (insertOp.ht().getType().cast<pgx::mlir::dsa::AggregationHashtableType>().getKeyType() == mlir::TupleType::get(getContext())) {
         auto loaded = rewriter.create<pgx::mlir::util::LoadOp>(loc, adaptor.ht().getType().cast<pgx::mlir::util::RefType>().getElementType(), adaptor.ht(), mlir::Value());
         auto newAggr = reduceFnBuilder(rewriter, loaded, adaptor.val());
         rewriter.create<pgx::mlir::util::StoreOp>(loc, newAggr, adaptor.ht(), mlir::Value());
         rewriter.eraseOp(insertOp);
      } else {
         Value hashed;
         {
            Block* sortLambda = &insertOp.hash().front();
            auto* sortLambdaTerminator = sortLambda->getTerminator();
            mlir::IRMapping mapping;
            mapping.map(sortLambda->getArgument(0), adaptor.key());

            for (auto& op : sortLambda->getOperations()) {
               if (&op != sortLambdaTerminator) {
                  rewriter.clone(op, mapping);
               }
            }
            hashed = mapping.lookup(cast<pgx::mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
         }

         auto keyType = adaptor.key().getType();
         auto aggrType = typeConverter->convertType(insertOp.ht().getType().cast<pgx::mlir::dsa::AggregationHashtableType>().getValType());
         auto* context = rewriter.getContext();
         auto entryType = getHashtableEntryType(context, keyType, aggrType);
         auto i8PtrType = pgx::mlir::util::RefType::get(context, IntegerType::get(context, 8));
         auto idxType = rewriter.getIndexType();
         auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
         auto kvType = getHashtableKVType(context, keyType, aggrType);
         auto kvPtrType = pgx::mlir::util::RefType::get(context, kvType);
         auto keyPtrType = pgx::mlir::util::RefType::get(context, keyType);
         auto aggrPtrType = pgx::mlir::util::RefType::get(context, aggrType);
         auto valuesType = pgx::mlir::util::RefType::get(context, entryType);
         auto entryPtrType = pgx::mlir::util::RefType::get(context, entryType);
         auto htType = pgx::mlir::util::RefType::get(context, entryPtrType);

         Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 1);
         Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 2);

         Value len = rewriter.create<util::LoadOp>(loc, idxType, lenAddress);
         Value capacityInitial = rewriter.create<util::LoadOp>(loc, idxType, capacityAddress);

         Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
         Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);

         Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacityInitial);
         rewriter.create<scf::IfOp>(
            loc, TypeRange(), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
               rt::Hashtable::resize(b,loc)(adaptor.ht());
               b.create<scf::YieldOp>(loc); });

         Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, pgx::mlir::util::RefType::get(rewriter.getContext(), htType), adaptor.ht(), 0);
         Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);

         Value capacity = rewriter.create<util::LoadOp>(loc, idxType, capacityAddress);

         Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         Value htSize = rewriter.create<arith::MulIOp>(loc, capacity, two);

         Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);
         //position = hash & hashTableMask
         Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
         // ptr = &hashtable[position]
         Type bucketPtrType = util::RefType::get(context, entryType);
         Type ptrType = util::RefType::get(context, bucketPtrType);
         Type doneType = rewriter.getI1Type();
         Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, ptrType, ht, position);

         auto resultTypes = std::vector<Type>({ptrType});
         auto whileOp = rewriter.create<scf::WhileOp>(loc, resultTypes, ValueRange({ptr}));
         Block* before = rewriter.createBlock(&whileOp.getBefore(), {}, resultTypes, {loc});
         Block* after = rewriter.createBlock(&whileOp.getAfter(), {}, resultTypes, {loc});

         // The conditional block of the while loop.
         {
            rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
            Value ptr = before->getArgument(0);

            Value currEntryPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ptr);
            //    if (*ptr != nullptr){
            Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
            auto ifOp = rewriter.create<scf::IfOp>(
               loc, TypeRange({doneType, ptrType}), cmp,
               [&](OpBuilder& b, Location loc) {

                  Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
                  Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                  auto ifOpH = b.create<scf::IfOp>(
                     loc, TypeRange({doneType,ptrType}), hashMatches, [&](OpBuilder& b, Location loc) {
                        Value kvAddress=rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                        Value entryKeyAddress=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                        Value entryKey = rewriter.create<util::LoadOp>(loc, keyType, entryKeyAddress);

                        Value keyMatches = equalFnBuilder(b,entryKey,adaptor.key());
                        auto ifOp2 = b.create<scf::IfOp>(
                           loc, TypeRange({doneType,ptrType}), keyMatches, [&](OpBuilder& b, Location loc) {
                              //          entry.aggr = update(vec.aggr,val)
                              if(reduceFnBuilder) {
                                 Value entryAggrAddress = rewriter.create<util::TupleElementPtrOp>(loc, aggrPtrType, kvAddress, 1);
                                 Value entryAggr = rewriter.create<util::LoadOp>(loc, aggrType, entryAggrAddress);
                                 Value newAggr = reduceFnBuilder(b, entryAggr, adaptor.val());
                                 b.create<util::StoreOp>(loc, newAggr, entryAggrAddress, Value());
                              }
                              b.create<scf::YieldOp>(loc, ValueRange{falseValue,ptr});
                           }, [&](OpBuilder& b, Location loc) {

                              //          ptr = &entry.next
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, ptrType,currEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
                        b.create<scf::YieldOp>(loc, ifOp2.getResults());
                     }, [&](OpBuilder& b, Location loc) {

                        //          ptr = &entry.next
                        Value newPtr=b.create<util::GenericMemrefCastOp>(loc, ptrType,currEntryPtr);
                        //          yield ptr,done=false
                        b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
                  b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
                  Value initValAddress = rewriter.create<util::TupleElementPtrOp>(loc, pgx::mlir::util::RefType::get(rewriter.getContext(), aggrType), adaptor.ht(), 5);
                  Value initialVal = b.create<util::LoadOp>(loc, aggrType, initValAddress);
                  Value newAggr = reduceFnBuilder ? reduceFnBuilder(b,initialVal, adaptor.val()): initialVal;
                  Value newKVPair = b.create<util::PackOp>(loc,ValueRange({adaptor.key(), newAggr}));
                  Value invalidNext  = b.create<util::InvalidRefOp>(loc,i8PtrType);
                  //       %newEntry = ...
                  Value newEntry = b.create<util::PackOp>(loc, ValueRange({invalidNext, hashed, newKVPair}));
                  Value valuesAddress = b.create<util::TupleElementPtrOp>(loc, pgx::mlir::util::RefType::get(b.getContext(),valuesType), adaptor.ht(), 3);
                  Value values = b.create<util::LoadOp>(loc, valuesType, valuesAddress);
                  Value newValueLocPtr=b.create<util::ArrayElementPtrOp>(loc,bucketPtrType,values,len);
                  //       append(vec,newEntry)
                  b.create<util::StoreOp>(loc, newEntry, newValueLocPtr,Value());

                  //       *ptr=len
                  b.create<util::StoreOp>(loc, newValueLocPtr, ptr, Value());
                  Value newLen = b.create<arith::AddIOp>(loc, len, one);
                  //       yield 0,0,done=true
                  b.create<pgx::mlir::util::StoreOp>(loc, newLen, lenAddress, Value());

                  b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });
            //       if(compare(entry.key,key)){

            Value done = ifOp.getResult(0);
            Value newPtr = ifOp.getResult(1);
            rewriter.create<scf::ConditionOp>(loc, done,
                                              ValueRange({newPtr}));
         }

         // The body of the while loop: shift right until reaching a value of 0.
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
class LazyJHtInsertLowering : public OpConversionPattern<pgx::mlir::dsa::HashtableInsert> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::HashtableInsert>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::HashtableInsert insertOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!insertOp.ht().getType().isa<pgx::mlir::dsa::JoinHashtableType>()) {
         return failure();
      }
      Value hashed;
      {
         Block* sortLambda = &insertOp.hash().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::IRMapping mapping;
         mapping.map(sortLambda->getArgument(0), adaptor.key());

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         hashed = mapping.lookup(cast<pgx::mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
      }
      mlir::Value val = adaptor.val();
      auto loc = insertOp->getLoc();

      if (!val) {
         val = rewriter.create<pgx::mlir::util::UndefOp>(loc, mlir::TupleType::get(getContext()));
      }
      auto entry = rewriter.create<pgx::mlir::util::PackOp>(loc, mlir::ValueRange({adaptor.key(), val}));
      auto bucket = rewriter.create<pgx::mlir::util::PackOp>(loc, mlir::ValueRange({hashed, entry}));
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto valuesType = pgx::mlir::util::RefType::get(rewriter.getContext(), bucket.getType());
      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 2);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 3);

      auto len = rewriter.create<pgx::mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = rewriter.create<pgx::mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      rewriter.create<scf::IfOp>(
         loc, TypeRange({}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            rt::LazyJoinHashtable::resize(b,loc)(adaptor.ht());
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = rewriter.create<util::TupleElementPtrOp>(loc, pgx::mlir::util::RefType::get(getContext(), adaptor.ht().getType().cast<pgx::mlir::util::RefType>().getElementType().cast<mlir::TupleType>().getType(4)), adaptor.ht(), 4);
      Value castedValuesAddress = rewriter.create<pgx::mlir::util::GenericMemrefCastOp>(loc, pgx::mlir::util::RefType::get(getContext(), valuesType), valuesAddress);
      auto values = rewriter.create<pgx::mlir::util::LoadOp>(loc, valuesType, castedValuesAddress, Value());
      rewriter.create<util::StoreOp>(loc, bucket, values, len);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = rewriter.create<arith::AddIOp>(loc, len, one);

      rewriter.create<pgx::mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
      rewriter.eraseOp(insertOp);
      return success();
   }
};
class FinalizeLowering : public OpConversionPattern<pgx::mlir::dsa::Finalize> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::Finalize>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::Finalize finalizeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!finalizeOp.ht().getType().isa<pgx::mlir::dsa::JoinHashtableType>()) {
         return failure();
      }
      rt::LazyJoinHashtable::finalize(rewriter, finalizeOp->getLoc())(adaptor.ht());
      rewriter.eraseOp(finalizeOp);
      return success();
   }
};
class FinalizeTBLowering : public OpConversionPattern<pgx::mlir::dsa::Finalize> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::Finalize>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::Finalize finalizeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!finalizeOp.ht().getType().isa<pgx::mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      mlir::Value res = rt::TableBuilder::build(rewriter, finalizeOp->getLoc())(adaptor.ht())[0];
      rewriter.replaceOp(finalizeOp, res);
      return success();
   }
};
class DSAppendLowering : public OpConversionPattern<pgx::mlir::dsa::Append> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!appendOp.ds().getType().isa<pgx::mlir::dsa::VectorType>()) {
         return failure();
      }
      Value builderVal = adaptor.ds();
      Value v = adaptor.val();
      auto convertedElementType = typeConverter->convertType(appendOp.ds().getType().cast<pgx::mlir::dsa::VectorType>().getElementType());
      auto loc = appendOp->getLoc();
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto valuesType = pgx::mlir::util::RefType::get(rewriter.getContext(), convertedElementType);
      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, builderVal, 0);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, builderVal, 1);

      auto len = rewriter.create<pgx::mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = rewriter.create<pgx::mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      rewriter.create<scf::IfOp>(
         loc, TypeRange({}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            rt::Vector::resize(b,loc)({builderVal});
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), valuesType), builderVal, 2);
      auto values = rewriter.create<pgx::mlir::util::LoadOp>(loc, valuesType, valuesAddress, Value());

      rewriter.create<util::StoreOp>(loc, v, values, len);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = rewriter.create<arith::AddIOp>(loc, len, one);
      rewriter.create<pgx::mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
      rewriter.eraseOp(appendOp);
      return success();
   }
};

class TBAppendLowering : public OpConversionPattern<pgx::mlir::dsa::Append> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!appendOp.ds().getType().isa<pgx::mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      Value builderVal = adaptor.ds();
      Value val = adaptor.val();
      Value isValid = adaptor.valid();
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
            case 128: rt::TableBuilder::addDecimal(rewriter, loc)({builderVal, isValid, val}); break;
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
      } else if (auto stringType = type.dyn_cast_or_null<pgx::mlir::util::VarLen32Type>()) {
         rt::TableBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      }
      rewriter.eraseOp(appendOp);
      return success();
   }
};
class NextRowLowering : public OpConversionPattern<pgx::mlir::dsa::NextRow> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::NextRow>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::NextRow op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::TableBuilder::nextRow(rewriter, op->getLoc())({adaptor.builder()});
      rewriter.eraseOp(op);
      return success();
   }
};
class CreateTableBuilderLowering : public OpConversionPattern<pgx::mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.ds().getType().isa<pgx::mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value schema = rewriter.create<pgx::mlir::util::CreateConstVarLen>(loc, pgx::mlir::util::VarLen32Type::get(getContext()), createOp.init_attr().getValue().cast<StringAttr>().str());
      Value tableBuilder = rt::TableBuilder::create(rewriter, loc)({schema})[0];
      rewriter.replaceOp(createOp, tableBuilder);
      return success();
   }
};
class FreeLowering : public OpConversionPattern<pgx::mlir::dsa::FreeOp> {
   public:
   using OpConversionPattern<pgx::mlir::dsa::FreeOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::dsa::FreeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto aggrHtType = op.val().getType().dyn_cast<pgx::mlir::dsa::AggregationHashtableType>()) {
         if (aggrHtType.getKeyType().getTypes().empty()) {
         } else {
            rt::Hashtable::destroy(rewriter, op->getLoc())(ValueRange{adaptor.val()});
         }
      }
      if (op.val().getType().isa<pgx::mlir::dsa::JoinHashtableType>()) {
         rt::LazyJoinHashtable::destroy(rewriter, op->getLoc())(ValueRange{adaptor.val()});
      }
      if (op.val().getType().isa<pgx::mlir::dsa::VectorType>()) {
         rt::Vector::destroy(rewriter, op->getLoc())(ValueRange{adaptor.val()});
      }
      rewriter.eraseOp(op);
      return success();
   }
};

} // end namespace
namespace pgx::mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateDsLowering, HtInsertLowering, FinalizeLowering, DSAppendLowering, LazyJHtInsertLowering, FreeLowering>(typeConverter, patterns.getContext());
   patterns.insert<CreateTableBuilderLowering, TBAppendLowering, FinalizeTBLowering, NextRowLowering>(typeConverter, patterns.getContext());
   typeConverter.addConversion([&typeConverter](pgx::mlir::dsa::VectorType vectorType) {
      return getLoweredVectorType(vectorType.getContext(), typeConverter.convertType(vectorType.getElementType()));
   });
   typeConverter.addConversion([&typeConverter](pgx::mlir::dsa::AggregationHashtableType aggregationHashtableType) {
      if (aggregationHashtableType.getKeyType().getTypes().empty()) {
         return (Type) pgx::mlir::util::RefType::get(aggregationHashtableType.getContext(), typeConverter.convertType(aggregationHashtableType.getValType()));
      } else {
         return (Type) getHashtableType(aggregationHashtableType.getContext(), typeConverter.convertType(aggregationHashtableType.getKeyType()), typeConverter.convertType(aggregationHashtableType.getValType()));
      }
   });
}
} // end namespace pgx::mlir::dsa
