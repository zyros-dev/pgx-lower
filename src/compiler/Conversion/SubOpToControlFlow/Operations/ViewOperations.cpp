#include "../Headers/SubOpToControlFlowPatterns.h"
#include "../Headers/SubOpToControlFlowUtilities.h"

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// Using declarations for utility functions
using subop_to_control_flow::implementBufferIteration;
using subop_to_control_flow::implementBufferIterationRuntime;
using subop_to_control_flow::getHtKVType;
using subop_to_control_flow::getHtEntryType;
using subop_to_control_flow::getHashMultiMapEntryType;
using subop_to_control_flow::getHashMultiMapValueType;
using subop_to_control_flow::hashKeys;
using subop_to_control_flow::unpackTypes;
using subop_to_control_flow::inlineBlock;

//===----------------------------------------------------------------------===//
// View Operations - Sorting, References, and Range Operations
//===----------------------------------------------------------------------===//

/// Lowering for CreateSortedViewOp - creates a sorted view of buffer data
class SortLowering : public SubOpConversionPattern<subop::CreateSortedViewOp> {
   public:
   using SubOpConversionPattern<subop::CreateSortedViewOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateSortedViewOp sortOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      PGX_DEBUG("SortLowering: Processing CreateSortedViewOp");
      
      static size_t id = 0;
      auto bufferType = mlir::cast<subop::BufferType>(sortOp.getToSort().getType());
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      EntryStorageHelper storageHelper(sortOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      ModuleOp parentModule = sortOp->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;

      // Create comparison function in module
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "sort_compare" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
      });
      
      auto* funcBody = new Block;
      funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
      funcOp.getBody().push_back(funcBody);
      
      // Implement comparison logic
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto leftVals = storageHelper.getValueMap(funcBody->getArgument(0), rewriter, sortOp->getLoc(), sortOp.getSortBy());
         auto rightVals = storageHelper.getValueMap(funcBody->getArgument(1), rewriter, sortOp->getLoc(), sortOp.getSortBy());
         std::vector<mlir::Value> args;
         for (const auto& pair : leftVals) {
             args.push_back(pair.second);
         }
         for (const auto& pair : rightVals) {
             args.push_back(pair.second);
         }
         Block* sortLambda = &sortOp.getRegion().front();
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
            rewriter.create<mlir::func::ReturnOp>(sortOp->getLoc(), adaptor.getResults());
         });
      });

      // Create sorted buffer using runtime function
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(sortOp->getLoc(), funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      auto genericBuffer = rt::GrowingBuffer::sort(rewriter, sortOp->getLoc())({adaptor.getToSort(), functionPointer})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(sortOp, typeConverter->convertType(sortOp.getType()), genericBuffer);
      
      PGX_DEBUG("SortLowering: Successfully created sorted view");
      return mlir::success();
   }
};

/// Lowering for GetBeginReferenceOp - gets reference to beginning of buffer
class GetBeginLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GetBeginReferenceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GetBeginReferenceOp>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::GetBeginReferenceOp getBeginReferenceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      PGX_DEBUG("GetBeginLowering: Creating begin reference");
      
      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(getBeginReferenceOp->getLoc(), 0);
      auto packed = rewriter.create<util::PackOp>(getBeginReferenceOp->getLoc(), mlir::ValueRange{zero, adaptor.getState()});
      mapping.define(getBeginReferenceOp.getRef(), packed);
      rewriter.replaceTupleStream(getBeginReferenceOp, mapping);
      return success();
   }
};

/// Lowering for GetEndReferenceOp - gets reference to end of buffer
class GetEndLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GetEndReferenceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GetEndReferenceOp>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::GetEndReferenceOp getEndReferenceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      PGX_DEBUG("GetEndLowering: Creating end reference");
      
      auto len = rewriter.create<util::BufferGetLen>(getEndReferenceOp->getLoc(), rewriter.getIndexType(), adaptor.getState());
      auto one = rewriter.create<mlir::arith::ConstantIndexOp>(getEndReferenceOp->getLoc(), 1);
      auto lastOffset = rewriter.create<mlir::arith::SubIOp>(getEndReferenceOp->getLoc(), len, one);
      auto packed = rewriter.create<util::PackOp>(getEndReferenceOp->getLoc(), mlir::ValueRange{lastOffset, adaptor.getState()});
      mapping.define(getEndReferenceOp.getRef(), packed);
      rewriter.replaceTupleStream(getEndReferenceOp, mapping);
      return success();
   }
};

/// Lowering for EntriesBetweenOp - calculates number of entries between two references
class EntriesBetweenLowering : public SubOpTupleStreamConsumerConversionPattern<subop::EntriesBetweenOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::EntriesBetweenOp>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::EntriesBetweenOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      PGX_DEBUG("EntriesBetweenLowering: Calculating entries between references");
      
      llvm::SmallVector<mlir::Value> unpackedLeft;
      llvm::SmallVector<mlir::Value> unpackedRight;
      rewriter.createOrFold<util::UnPackOp>(unpackedLeft, op->getLoc(), mapping.resolve(op, op.getLeftRef()));
      rewriter.createOrFold<util::UnPackOp>(unpackedRight, op->getLoc(), mapping.resolve(op, op.getRightRef()));
      
      mlir::Value difference = rewriter.create<mlir::arith::SubIOp>(op->getLoc(), unpackedRight[0], unpackedLeft[0]);
      if (!op.getBetween().getColumn().type.isIndex()) {
         difference = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), op.getBetween().getColumn().type, difference);
      }
      
      mapping.define(op.getBetween(), difference);
      rewriter.replaceTupleStream(op, mapping);
      return success();
   }
};

/// Lowering for OffsetReferenceBy - offsets a reference by a given amount with bounds checking
class OffsetReferenceByLowering : public SubOpTupleStreamConsumerConversionPattern<subop::OffsetReferenceBy> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::OffsetReferenceBy>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::OffsetReferenceBy op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      PGX_DEBUG("OffsetReferenceByLowering: Offsetting reference with bounds checking");
      
      llvm::SmallVector<mlir::Value> unpackedRef;
      rewriter.createOrFold<util::UnPackOp>(unpackedRef, op->getLoc(), mapping.resolve(op, op.getRef()));
      auto offset = mapping.resolve(op, op.getIdx());
      
      // Convert offset to index type if needed
      if (!offset.getType().isIndex()) {
         offset = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getIndexType(), offset);
      }
      
      // Perform arithmetic in i64 for safety
      offset = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getI64Type(), offset);
      auto currIdx = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getI64Type(), unpackedRef[0]);
      mlir::Value newIdx = rewriter.create<mlir::arith::AddIOp>(op->getLoc(), currIdx, offset);
      
      // Ensure non-negative
      newIdx = rewriter.create<mlir::arith::MaxSIOp>(op->getLoc(), rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, rewriter.getI64Type()), newIdx);
      newIdx = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getIndexType(), newIdx);
      
      // Ensure within buffer bounds
      auto length = rewriter.create<util::BufferGetLen>(op->getLoc(), rewriter.getIndexType(), unpackedRef[1]);
      auto maxIdx = rewriter.create<mlir::arith::SubIOp>(op->getLoc(), length, rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 1));
      newIdx = rewriter.create<mlir::arith::MinUIOp>(op->getLoc(), maxIdx, newIdx);

      auto newRef = rewriter.create<util::PackOp>(op->getLoc(), mlir::ValueRange{newIdx, unpackedRef[1]});
      mapping.define(op.getNewRef(), newRef);
      rewriter.replaceTupleStream(op, mapping);
      return success();
   }
};

/// Lowering for UnwrapOptionalRefOp when the optional contains a hashmap reference
class UnwrapOptionalHashmapRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::UnwrapOptionalRefOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      PGX_DEBUG("UnwrapOptionalHashmapRefLowering: Unwrapping optional hashmap reference");
      
      auto optionalType = mlir::dyn_cast_or_null<subop::OptionalType>(op.getOptionalRef().getColumn().type);
      if (!optionalType) return mlir::failure();
      auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(optionalType.getT());
      if (!lookupRefType) return mlir::failure();
      auto hashmapType = mlir::dyn_cast_or_null<subop::HashMapType>(lookupRefType.getState());
      if (!hashmapType) return mlir::failure();
      
      auto loc = op.getLoc();
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), mapping.resolve(op, op.getOptionalRef()));
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      
      // Set up type information for hashmap entry access
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      EntryStorageHelper valStorageHelper(op, hashmapType.getValueMembers(), hashmapType.hasLock(), typeConverter);
      auto valPtrType = valStorageHelper.getRefType();
      
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, op->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         // Navigate through hashmap entry structure to get value pointer
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, mapping.resolve(op, op.getOptionalRef()));
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
         mapping.define(op.getRef(), valuePtr);
         rewriter.replaceTupleStream(op, mapping);
      });
      
      PGX_DEBUG("UnwrapOptionalHashmapRefLowering: Successfully unwrapped hashmap reference");
      return mlir::success();
   }
};

/// Lowering for UnwrapOptionalRefOp when the optional contains a pre-aggregation hashtable reference
class UnwrapOptionalPreAggregationHtRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::UnwrapOptionalRefOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      PGX_DEBUG("UnwrapOptionalPreAggregationHtRefLowering: Unwrapping optional pre-aggregation hashtable reference");
      
      auto optionalType = mlir::dyn_cast_or_null<subop::OptionalType>(op.getOptionalRef().getColumn().type);
      if (!optionalType) return mlir::failure();
      auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(optionalType.getT());
      if (!lookupRefType) return mlir::failure();
      auto hashmapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(lookupRefType.getState());
      if (!hashmapType) return mlir::failure();
      
      auto loc = op.getLoc();
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), mapping.resolve(op, op.getOptionalRef()));
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      
      // Set up type information for pre-aggregation hashtable entry access
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      EntryStorageHelper valStorageHelper(op, hashmapType.getValueMembers(), hashmapType.hasLock(), typeConverter);
      auto valPtrType = valStorageHelper.getRefType();
      
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, op->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         // Navigate through pre-aggregation hashtable entry structure to get value pointer
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, mapping.resolve(op, op.getOptionalRef()));
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
         mapping.define(op.getRef(), valuePtr);
         rewriter.replaceTupleStream(op, mapping);
      });
      
      PGX_DEBUG("UnwrapOptionalPreAggregationHtRefLowering: Successfully unwrapped pre-aggregation hashtable reference");
      return mlir::success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower