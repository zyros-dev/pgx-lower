#include "mlir/Conversion/DSAToStd/CollectionIteration.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "runtime-defs/Vector.h"
#include "execution/logging.h"
using namespace mlir;
namespace {

class SortOpLowering : public OpConversionPattern<mlir::dsa::SortOp> {
   public:
   using OpConversionPattern<mlir::dsa::SortOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::SortOp sortOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      static size_t id = 0;

      auto ptrType = mlir::util::RefType::get(getContext(), IntegerType::get(getContext(), 8));

      ModuleOp parentModule = sortOp->getParentOfType<ModuleOp>();
      Type elementType = llvm::cast<mlir::dsa::VectorType>(sortOp.getToSort().getType()).getElementType();
      ::mlir::func::FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto funcType = rewriter.getFunctionType(TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type()));
         funcOp = rewriter.create<::mlir::func::FuncOp>(parentModule.getLoc(), "dsa_sort_compare" + std::to_string(id++), funcType);
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
         funcOp.getBody().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value left = funcBody->getArgument(0);
         Value right = funcBody->getArgument(1);

         Value genericMemrefLeft = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), util::RefType::get(rewriter.getContext(), elementType), left);
         Value genericMemrefRight = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), util::RefType::get(rewriter.getContext(), elementType), right);
         Value tupleLeft = rewriter.create<util::LoadOp>(sortOp.getLoc(), elementType, genericMemrefLeft, Value());
         Value tupleRight = rewriter.create<util::LoadOp>(sortOp.getLoc(), elementType, genericMemrefRight, Value());
         auto terminator = rewriter.create<mlir::func::ReturnOp>(sortOp.getLoc());
         Block* sortLambda = &sortOp.getRegion().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.moveBlockBefore(sortLambda, terminator.getOperation()->getBlock());
         sortLambda->getArgument(0).replaceAllUsesWith(tupleLeft);
         sortLambda->getArgument(1).replaceAllUsesWith(tupleRight);
         mlir::dsa::YieldOp yieldOp = mlir::cast<mlir::dsa::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.getResults()[0];
         rewriter.create<mlir::func::ReturnOp>(sortOp.getLoc(), x);
         rewriter.eraseOp(sortLambdaTerminator);
         rewriter.eraseOp(terminator);
      }


      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(sortOp->getLoc(), funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      rt::Vector::sort(rewriter, sortOp->getLoc())({adaptor.getToSort(), functionPointer});
      rewriter.eraseOp(sortOp);
      return success();
   }
};
class ForOpLowering : public OpConversionPattern<mlir::dsa::ForOp> {
   public:
   using OpConversionPattern<mlir::dsa::ForOp>::OpConversionPattern;
   std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) const {
      for (size_t i = 0; i < values.size(); i++) {
         values[i] = builder.getRemappedValue(values[i]);
      }
      return values;
   }

   LogicalResult matchAndRewrite(mlir::dsa::ForOp forOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: ENTRY");
      
      // CRITICAL: Check if region is empty before accessing
      if (forOp.getRegion().empty()) {
         MLIR_PGX_ERROR("DSA", "ForOpLowering: ForOp has empty region!");
         return failure();
      }
      
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: ForOp has non-empty region");
      
      // CRITICAL: Capture the body block and terminator BEFORE they get moved
      Block* originalBody = forOp.getBody();
      mlir::dsa::YieldOp originalYieldOp = cast<mlir::dsa::YieldOp>(originalBody->getTerminator());
      
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: Captured original body and yield op");
      
      std::vector<Type> argumentTypes;
      std::vector<Location> argumentLocs;
      for (auto t : forOp.getRegion().getArgumentTypes()) {
         argumentTypes.push_back(t);
         argumentLocs.push_back(forOp->getLoc());
      }
      // CRITICAL: Pass the actual type to getImpl, don't cast to CollectionType first
      auto actualType = forOp.getCollection().getType();
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: Getting iterator for collection");
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: Original forOp collection type is valid");
      
      // Check if it's a GenericIterableType for debugging
      if (auto genIterType = actualType.dyn_cast_or_null<mlir::dsa::GenericIterableType>()) {
         MLIR_PGX_DEBUG("DSA", "ForOpLowering: Type is GenericIterableType with name: " + genIterType.getIteratorName());
      } else {
         MLIR_PGX_ERROR("DSA", "ForOpLowering: actualType is NOT a GenericIterableType!");
         // Try to understand what type it is
         if (auto recordBatchType = actualType.dyn_cast_or_null<mlir::dsa::RecordBatchType>()) {
            MLIR_PGX_ERROR("DSA", "ForOpLowering: actualType is RecordBatchType");
         } else {
            MLIR_PGX_ERROR("DSA", "ForOpLowering: actualType is completely unknown type");
         }
      }
      
      // CRITICAL: During conversion, the adaptor's collection value is already converted to i8*
      // We MUST use the ORIGINAL type from forOp, not the converted value's type
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: Using original ForOp collection type for iterator lookup");
      
      auto iterator = mlir::dsa::CollectionIterationImpl::getImpl(actualType, adaptor.getCollection());
      if (!iterator) {
         MLIR_PGX_ERROR("DSA", "ForOpLowering: Failed to get iterator for collection type!");
         return failure();
      }
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: Got iterator successfully");

      ModuleOp parentModule = forOp->getParentOfType<ModuleOp>();
      bool containsCondSkip = false;
      for (auto& op : forOp.getRegion().front().getOperations()) {
         containsCondSkip |= mlir::isa<mlir::dsa::CondSkipOp>(&op);
      }

      using fn_t = std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)>;
      fn_t fn1 = [&, originalBody, originalYieldOp](std::function<Value(OpBuilder & b)> getElem, ValueRange iterargs, OpBuilder builder) {
         // Use captured values instead of accessing forOp.getBody()
         auto yieldOp = originalYieldOp;
         std::vector<Type> resTypes;
         std::vector<Location> locs;
         for (auto t : yieldOp.getResults()) {
            resTypes.push_back(typeConverter->convertType(t.getType()));
            locs.push_back(forOp->getLoc());
         }
         std::vector<Value> values;
         values.push_back(getElem(builder));
         values.insert(values.end(), iterargs.begin(), iterargs.end());
         auto execRegion = builder.create<mlir::scf::ExecuteRegionOp>(forOp->getLoc(), resTypes);
         auto* execRegionBlock = new Block();
         execRegion.getRegion().push_back(execRegionBlock);
         {
            OpBuilder::InsertionGuard guard(builder);
            OpBuilder::InsertionGuard guard2(rewriter);

            builder.setInsertionPointToStart(execRegionBlock);

            auto term = builder.create<mlir::scf::YieldOp>(forOp->getLoc());
            builder.setInsertionPoint(term);
            rewriter.moveBlockBefore(originalBody, builder.getInsertionBlock());
            originalBody->getArguments().front().replaceAllUsesWith(values[0]);

            std::vector<Value> results(yieldOp.getResults().begin(), yieldOp.getResults().end());
            rewriter.eraseOp(yieldOp);
            auto term2 = builder.create<mlir::scf::YieldOp>(forOp->getLoc(), remap(results, rewriter));

            Block* end = rewriter.splitBlock(builder.getBlock(), builder.getInsertionPoint());
            builder.setInsertionPoint(term2);
            builder.create<mlir::cf::BranchOp>(forOp->getLoc(), end, remap(results, rewriter));
            rewriter.eraseOp(term2);

            end->addArguments(resTypes, locs);
            builder.setInsertionPointToEnd(end);
            builder.create<mlir::scf::YieldOp>(forOp->getLoc(), end->getArguments());
            std::vector<Operation*> toErase;
            for (auto it = execRegionBlock->getOperations().rbegin(); it != execRegionBlock->getOperations().rend(); it++) {
               if (auto op = mlir::dyn_cast_or_null<mlir::dsa::CondSkipOp>(&*it)) {
                  toErase.push_back(op.getOperation());
                  builder.setInsertionPointAfter(op);
                  llvm::SmallVector<::mlir::Value> remappedArgs;
                  assert(rewriter.getRemappedValues(op.getArgs(), remappedArgs).succeeded());
                  Block* after = rewriter.splitBlock(builder.getBlock(), builder.getInsertionPoint());
                  builder.setInsertionPointAfter(op);
                  auto cond = rewriter.getRemappedValue(op.getCondition());
                  builder.create<mlir::cf::CondBranchOp>(op->getLoc(), cond, end, remappedArgs, after, ValueRange());
               }
            }
            for (auto* x : toErase) {
               rewriter.eraseOp(x);
            }
            rewriter.eraseOp(term);
         }
         //yieldOp->erase();
         assert(execRegion.getNumResults() == resTypes.size());
         std::vector<Value> results(execRegion.getResults().begin(), execRegion.getResults().end());

         return results;
      };
      fn_t fn2 = [&, originalBody, originalYieldOp](std::function<Value(OpBuilder & b)> getElem, ValueRange iterargs, OpBuilder builder) {
         // Use captured values instead of accessing forOp.getBody()
         auto yieldOp = originalYieldOp;
         std::vector<Type> resTypes;
         std::vector<Location> locs;
         for (auto t : yieldOp.getResults()) {
            resTypes.push_back(typeConverter->convertType(t.getType()));
            locs.push_back(forOp->getLoc());
         }
         std::vector<Value> values;
         values.push_back(getElem(builder));
         values.insert(values.end(), iterargs.begin(), iterargs.end());
         auto term = builder.create<mlir::scf::YieldOp>(forOp->getLoc());
         builder.setInsertionPoint(term);
         rewriter.moveBlockBefore(originalBody, builder.getInsertionBlock());
         auto args = originalBody->getArguments();
         for (size_t i = 0; i < args.size() && i < values.size(); ++i) {
            args[i].replaceAllUsesWith(values[i]);
         }

         std::vector<Value> results(yieldOp.getResults().begin(), yieldOp.getResults().end());
         rewriter.eraseOp(yieldOp);
         rewriter.eraseOp(term);

         return results;
      };

      MLIR_PGX_DEBUG("DSA", "ForOpLowering: About to call implementLoop");
      std::vector<Value> results = iterator->implementLoop(forOp->getLoc(), adaptor.getInitArgs(), forOp.getUntil(), *typeConverter, rewriter, parentModule, containsCondSkip ? fn1 : fn2);
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: implementLoop returned successfully");
      
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);

         MLIR_PGX_DEBUG("DSA", "ForOpLowering: Replacing ForOp region with empty block");
         forOp.getRegion().push_back(new Block());
         forOp.getRegion().front().addArguments(argumentTypes, argumentLocs);
         rewriter.setInsertionPointToStart(&forOp.getRegion().front());
         rewriter.create<mlir::dsa::YieldOp>(forOp.getLoc());
      }

      MLIR_PGX_DEBUG("DSA", "ForOpLowering: Replacing ForOp with results");
      rewriter.replaceOp(forOp, results);
      MLIR_PGX_DEBUG("DSA", "ForOpLowering: SUCCESS");
      return success();
   }
};

class LookupOpLowering  : public OpConversionPattern<mlir::dsa::Lookup> {
   public:
   using OpConversionPattern<mlir::dsa::Lookup>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Lookup op, OpAdaptor lookupAdaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto loaded = rewriter.create<util::LoadOp>(loc, llvm::cast<mlir::util::RefType>(lookupAdaptor.getCollection().getType()).getElementType(), lookupAdaptor.getCollection(), Value());
      auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, loaded);
      Value ht = unpacked.getResult(0);
      Value htMask = unpacked.getResult(1);
      Value buckedPos = rewriter.create<arith::AndIOp>(loc, htMask, lookupAdaptor.getKey());
      Value ptr = rewriter.create<util::LoadOp>(loc, llvm::cast<mlir::util::RefType>(typeConverter->convertType(ht.getType())).getElementType(), ht, buckedPos);
      //optimization
      ptr = rewriter.create<mlir::util::FilterTaggedPtr>(loc, ptr.getType(), ptr, lookupAdaptor.getKey());
      Value packed = rewriter.create<mlir::util::PackOp>(loc, ValueRange{ptr, lookupAdaptor.getKey()});
      rewriter.replaceOp(op, packed);
      return success();
   }
};
class AtLowering  : public OpConversionPattern<mlir::dsa::At> {
   public:
   using OpConversionPattern<mlir::dsa::At>::OpConversionPattern;
   static Value getBit(OpBuilder builder, Location loc, Value bits, Value pos) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 7));
      Value const1Byte = builder.create<arith::ConstantOp>(loc, i8Type, builder.getIntegerAttr(i8Type, 1));

      Value div8 = builder.create<arith::ShRUIOp>(loc, indexType, pos, const3);
      Value rem8 = builder.create<arith::AndIOp>(loc, indexType, pos, const7);
      Value loadedByte = builder.create<mlir::util::LoadOp>(loc, i8Type, bits, div8);
      Value rem8AsByte = builder.create<arith::IndexCastOp>(loc, i8Type, rem8);
      Value shifted = builder.create<arith::ShRUIOp>(loc, i8Type, loadedByte, rem8AsByte);
      Value res1 = shifted;

      Value anded = builder.create<arith::AndIOp>(loc, i8Type, res1, const1Byte);
      Value res = builder.create<arith::CmpIOp>(loc, i1Type, mlir::arith::CmpIPredicate::eq, anded, const1Byte);
      return res;
   }

   public:
   LogicalResult matchAndRewrite(mlir::dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = atOp->getLoc();
      auto baseType = getBaseType(atOp.getType(0));
      ::mlir::Value index;
      ::mlir::Value columnOffset;
      auto indexType = rewriter.getIndexType();
      ::mlir::Value originalValueBuffer;
      ::mlir::Value valueBuffer;
      ::mlir::Value validityBuffer;
      ::mlir::Value varLenBuffer;
      ::mlir::Value nullMultiplier;
      {
         ::mlir::OpBuilder::InsertionGuard guard(rewriter);
         if (auto* definingOp = adaptor.getCollection().getDefiningOp()) {
            rewriter.setInsertionPointAfter(definingOp);
         }
         auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getCollection());
         index = unpacked.getResult(0);
         auto info = unpacked.getResult(1);
         size_t column = atOp.getPos();
         size_t baseOffset = 1 + column * 5;
         columnOffset = rewriter.create<mlir::util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset);
         validityBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, llvm::cast<TupleType>(info.getType()).getType(baseOffset + 2), info, baseOffset + 2);
         originalValueBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, llvm::cast<TupleType>(info.getType()).getType(baseOffset + 3), info, baseOffset + 3);
         valueBuffer = rewriter.create<mlir::util::ArrayElementPtrOp>(loc, originalValueBuffer.getType(), originalValueBuffer, columnOffset);
         varLenBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, llvm::cast<TupleType>(info.getType()).getType(baseOffset + 4), info, baseOffset + 4);
         nullMultiplier = rewriter.create<mlir::util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset + 1);
      }
      Value val;
      auto* context = rewriter.getContext();
      if (baseType.isa<util::VarLen32Type>()) {
         Value pos1 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, index);
         pos1.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value const1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         Value ip1 = rewriter.create<arith::AddIOp>(loc, indexType, index, const1);
         Value pos2 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, ip1);
         pos2.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value len = rewriter.create<arith::SubIOp>(loc, rewriter.getI32Type(), pos2, pos1);
         Value pos1AsIndex = rewriter.create<arith::IndexCastOp>(loc, indexType, pos1);
         Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), varLenBuffer, pos1AsIndex);
         val = rewriter.create<mlir::util::CreateVarLen>(loc, mlir::util::VarLen32Type::get(rewriter.getContext()), ptr, len);
      } else if (isIntegerType(baseType, 1)) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         val = getBit(rewriter, loc, originalValueBuffer, realPos);
      } else if (typeConverter->convertType(baseType).isIntOrIndexOrFloat()) {
         auto convertedType = typeConverter->convertType(baseType);
         if (convertedType.isInteger(24) || convertedType.isInteger(48) || convertedType.isInteger(56)) {
            Value factor = rewriter.create<mlir::arith::ConstantIndexOp>(loc, llvm::cast<mlir::IntegerType>(convertedType).getWidth() / 8);
            Value pos = rewriter.create<arith::AddIOp>(loc, columnOffset, index);
            pos = rewriter.create<arith::MulIOp>(loc, pos, factor);
            Value valBuffer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI8Type()), originalValueBuffer);
            Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos);
            ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, convertedType), ptr);
            val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), ptr);
            val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         } else {
            val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), valueBuffer, index);
            val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         }
      } else {
         assert(val && "unhandled type!!");
      }
      if (atOp->getNumResults() == 2) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         realPos = rewriter.create<arith::MulIOp>(loc, indexType, nullMultiplier, index);
         Value isValid = getBit(rewriter, loc, validityBuffer, realPos);
         rewriter.replaceOp(atOp, ::mlir::ValueRange{val, isValid});
      } else {
         rewriter.replaceOp(atOp, val);
      }
      return success();
   }
};
} // namespace

void mlir::dsa::populateCollectionsToStdPatterns(::mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto* context = patterns.getContext();

   patterns.insert<SortOpLowering>(typeConverter, context);

   patterns.insert<ForOpLowering>(typeConverter, context);
   patterns.insert<LookupOpLowering>(typeConverter, context);
   patterns.insert<AtLowering>(typeConverter, context);

   auto indexType = IndexType::get(context);
   auto i8ptrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
   auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));


   typeConverter.addConversion([&typeConverter, indexType, i8PtrType, context](mlir::dsa::JoinHashtableType joinHashtableType) {
      Type kvType = typeConverter.convertType(TupleType::get(context, {joinHashtableType.getKeyType(), joinHashtableType.getValType()}));
      Type entryType = TupleType::get(context, {i8PtrType, kvType});

      auto vecType = mlir::util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, mlir::util::RefType::get(context, entryType));
      return (Type) util::RefType::get(context, TupleType::get(context, {htType, indexType, indexType, indexType, vecType}));
   });


   typeConverter.addConversion([context, i8ptrType, indexType](mlir::dsa::RecordBatchType recordBatchType) {
      std::vector<Type> types;
      types.push_back(indexType);
      if (auto tupleT = recordBatchType.getRowType().dyn_cast_or_null<TupleType>()) {
         for (auto t : tupleT.getTypes()) {
            if (t.isa<mlir::util::VarLen32Type>()) {
               t = mlir::IntegerType::get(context, 32);
            } else if (t == mlir::IntegerType::get(context, 1)) {
               t = mlir::IntegerType::get(context, 8);
            }

            types.push_back(indexType);
            types.push_back(indexType);
            types.push_back(i8ptrType);
            types.push_back(mlir::util::RefType::get(context, t));
            types.push_back(i8ptrType);
         }
      }
      return (Type) TupleType::get(context, types);
   });
   typeConverter.addConversion([context, &typeConverter, indexType](mlir::dsa::RecordType recordType) {
      return (Type) TupleType::get(context, {indexType, typeConverter.convertType(mlir::dsa::RecordBatchType::get(context, recordType.getRowType()))});
   });

   typeConverter.addConversion([&typeConverter, context, i8ptrType, indexType](mlir::dsa::GenericIterableType genericIterableType) {
      Type elementType = genericIterableType.getElementType();
      Type nestedElementType = elementType;
      if (auto nested = elementType.dyn_cast_or_null<mlir::dsa::GenericIterableType>()) {
         nestedElementType = nested.getElementType();
      }
      if (genericIterableType.getIteratorName() == "table_chunk_iterator") {
         return (Type) i8ptrType;
      } else if (genericIterableType.getIteratorName() == "join_ht_iterator") {
         auto ptrType = mlir::util::RefType::get(context, typeConverter.convertType(TupleType::get(context, {i8ptrType, genericIterableType.getElementType()})));
         return (Type) TupleType::get(context, {ptrType, indexType});
      } else if (genericIterableType.getIteratorName() == "join_ht_mod_iterator") {
         auto types = llvm::cast<mlir::TupleType>(genericIterableType.getElementType()).getTypes();
         auto ptrType = mlir::util::RefType::get(context, typeConverter.convertType(TupleType::get(context, {i8ptrType, types[0]})));
         return (Type) TupleType::get(context, {ptrType, indexType});
      }
      return Type();
   });
}