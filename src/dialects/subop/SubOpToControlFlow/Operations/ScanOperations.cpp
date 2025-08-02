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

// Using terminator utilities for defensive programming
using subop_to_control_flow::TerminatorUtils::ensureTerminator;
using subop_to_control_flow::TerminatorUtils::ensureForOpTermination;
using subop_to_control_flow::TerminatorUtils::createContextAppropriateTerminator;
using subop_to_control_flow::TerminatorUtils::reportTerminatorStatus;

// Using runtime call termination utilities for comprehensive safety
using subop_to_control_flow::RuntimeCallTermination::applyRuntimeCallSafetyToOperation;
using subop_to_control_flow::RuntimeCallTermination::ensureGrowingBufferCallTermination;
using subop_to_control_flow::RuntimeCallTermination::ensureHashtableCallTermination;

// ============================================================================
// Scan Operations - Vector, State, and View Scanning
// ============================================================================

class ScanRefsVectorLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(scanOp.getState().getType());
      if (!bufferType) return failure();
      ColumnMapping mapping;
      auto elementType = EntryStorageHelper(scanOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter).getStorageType();

      auto iterator = rt::GrowingBuffer::createIterator(rewriter, scanOp->getLoc())(adaptor.getState())[0];
      implementBufferIteration(scanOp->hasAttr("parallel"), iterator, elementType, scanOp->getLoc(), rewriter, *typeConverter, scanOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         mapping.define(scanOp.getRef(), ptr);
         rewriter.replaceTupleStream(scanOp, mapping);
         
         // Ensure runtime call termination for GrowingBuffer::createIterator
         subop_to_control_flow::RuntimeCallTermination::ensureGrowingBufferCallTermination(
             iterator.getDefiningOp(), rewriter, scanOp->getLoc());
      });
      return success();
   }
};

class ScanRefsSimpleStateLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::SimpleStateType>(scanOp.getState().getType())) return failure();
      ColumnMapping mapping;
      mapping.define(scanOp.getRef(), adaptor.getState());
      rewriter.replaceTupleStream(scanOp, mapping);
      return success();
   }
};

class ScanRefsSortedViewLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto sortedViewType = mlir::dyn_cast_or_null<subop::SortedViewType>(scanOp.getState().getType());
      if (!sortedViewType) return failure();
      ColumnMapping mapping;
      auto elementType = util::RefType::get(getContext(), EntryStorageHelper(scanOp, sortedViewType.getMembers(), sortedViewType.hasLock(), typeConverter).getStorageType());
      auto loc = scanOp->getLoc();
      auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), adaptor.getState());
      auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
      rewriter.atStartOf(forOp.getBody(), [&](SubOpRewriter& rewriter) {
         auto currElementPtr = rewriter.create<util::BufferGetElementRef>(loc, elementType, adaptor.getState(), forOp.getInductionVar());
         mapping.define(scanOp.getRef(), currElementPtr);
         rewriter.replaceTupleStream(scanOp, mapping);
      });
      
      // Systematic terminator validation after ForOp construction
      ensureForOpTermination(forOp, rewriter, loc);

      return success();
   }
};

class ScanRefsHeapLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto heapType = mlir::dyn_cast_or_null<subop::HeapType>(scanOp.getState().getType());
      if (!heapType) return failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      EntryStorageHelper storageHelper(scanOp, heapType.getMembers(), heapType.hasLock(), typeConverter);
      mlir::TupleType elementType = storageHelper.getStorageType();
      auto buffer = rt::Heap::getBuffer(rewriter, scanOp->getLoc())({adaptor.getState()})[0];
      auto castedBuffer = rewriter.create<util::BufferCastOp>(loc, util::BufferType::get(rewriter.getContext(), elementType), buffer);
      auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), castedBuffer);
      auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
      rewriter.atStartOf(forOp.getBody(), [&](SubOpRewriter& rewriter) {
         auto currElementPtr = rewriter.create<util::BufferGetElementRef>(loc, util::RefType::get(elementType), castedBuffer, forOp.getInductionVar());
         mapping.define(scanOp.getRef(), currElementPtr);
         rewriter.replaceTupleStream(scanOp, mapping);
      });

      return success();
   }
};

class ScanRefsContinuousViewLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ContinuousViewType, subop::ArrayType>(scanOp.getState().getType())) return failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      auto bufferType = mlir::cast<util::BufferType>(adaptor.getState().getType());
      mlir::Value typeSize = rewriter.create<util::SizeOfOp>(scanOp->getLoc(), rewriter.getIndexType(), typeConverter->convertType(bufferType.getT()));

      auto* ctxt = rewriter.getContext();
      ModuleOp parentModule = typeSize.getDefiningOp()->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;
      static size_t funcIds;
      auto ptrType = util::RefType::get(ctxt, IntegerType::get(ctxt, 8));
      auto plainBufferType = util::BufferType::get(ctxt, mlir::IntegerType::get(ctxt, 8));
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "scan_cv_func" + std::to_string(funcIds++), mlir::FunctionType::get(ctxt, TypeRange{plainBufferType, rewriter.getI64Type(), rewriter.getI64Type(), ptrType}, TypeRange()));
      });
      auto* funcBody = new Block;
      mlir::Value buffer = funcBody->addArgument(plainBufferType, loc);
      mlir::Value startPos = funcBody->addArgument(rewriter.getI64Type(), loc);
      mlir::Value endPos = funcBody->addArgument(rewriter.getI64Type(), loc);

      mlir::Value contextPtr = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto guard = rewriter.loadStepRequirements(contextPtr, typeConverter);
         auto castedBuffer = rewriter.create<util::BufferCastOp>(loc, bufferType, buffer);
         startPos = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), startPos);
         endPos = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), endPos);
         auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp = rewriter.create<mlir::scf::ForOp>(scanOp->getLoc(), startPos, endPos, one);
         mlir::Block* block = forOp.getBody();
         rewriter.atStartOf(block, [&](SubOpRewriter& rewriter) {
            auto pair = rewriter.create<util::PackOp>(loc, mlir::ValueRange{forOp.getInductionVar(), castedBuffer});
            mapping.define(scanOp.getRef(), pair);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::func::ReturnOp>(loc);
      });
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      Value parallelConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, scanOp->hasAttr("parallel"), rewriter.getI1Type());
      rt::Buffer::iterate(rewriter, loc)({parallelConst, adaptor.getState(), typeSize, functionPointer, rewriter.storeStepRequirements()});
      return mlir::success();

      return success();
   }
};

// ============================================================================
// Hash Map and Hash Table Scanning Operations
// ============================================================================

class ScanHashMapLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanRefsOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMapType>(scanRefsOp.getState().getType())) return failure();
      auto hashMapType = mlir::cast<subop::HashMapType>(scanRefsOp.getState().getType());
      ColumnMapping mapping;
      auto loc = scanRefsOp->getLoc();
      auto it = rt::Hashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashMapType, *typeConverter));
      implementBufferIteration(scanRefsOp->hasAttr("parallel"), it, getHtEntryType(hashMapType, *typeConverter), loc, rewriter, *typeConverter, scanRefsOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         mapping.define(scanRefsOp.getRef(), kvPtr);
         rewriter.replaceTupleStream(scanRefsOp, mapping);
         
         // Ensure runtime call termination for Hashtable::createIterator
         subop_to_control_flow::RuntimeCallTermination::ensureHashtableCallTermination(
             it.getDefiningOp(), rewriter, loc);
      });
      return success();
   }
};

class ScanPreAggrHtLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanRefsOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtType>(scanRefsOp.getState().getType())) return failure();
      auto hashMapType = mlir::cast<subop::PreAggrHtType>(scanRefsOp.getState().getType());
      ColumnMapping mapping;
      auto loc = scanRefsOp->getLoc();
      auto it = rt::PreAggregationHashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashMapType, *typeConverter));
      implementBufferIteration(scanRefsOp->hasAttr("parallel"), it, util::RefType::get(getContext(), getHtEntryType(hashMapType, *typeConverter)), loc, rewriter, *typeConverter, scanRefsOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         ptr = rewriter.create<util::LoadOp>(loc, ptr, mlir::Value());
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         mapping.define(scanRefsOp.getRef(), kvPtr);
         rewriter.replaceTupleStream(scanRefsOp, mapping);
         
         // Ensure runtime call termination for PreAggregationHashtable::createIterator
         subop_to_control_flow::RuntimeCallTermination::ensurePreAggregationHashtableCallTermination(
             it.getDefiningOp(), rewriter, loc);
      });
      return success();
   }
};

class ScanHashMultiMap : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanRefsOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(scanRefsOp.getState().getType());
      if (!hashMultiMapType) return failure();
      ColumnMapping mapping;
      auto loc = scanRefsOp->getLoc();
      auto it = rt::Hashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      EntryStorageHelper valStorageHelper(scanRefsOp, hashMultiMapType.getValueMembers(), false, typeConverter);
      EntryStorageHelper keyStorageHelper(scanRefsOp, hashMultiMapType.getKeyMembers(), hashMultiMapType.hasLock(), typeConverter);
      auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
      auto i8PtrPtrType = util::RefType::get(getContext(), i8PtrType);

      implementBufferIteration(scanRefsOp->hasAttr("parallel"), it, getHashMultiMapEntryType(hashMultiMapType, *typeConverter), loc, rewriter, *typeConverter, scanRefsOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         auto keyPtr = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ptr, 3);
         auto valueListPtr = rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, ptr, 2);
         mlir::Value valuePtr = rewriter.create<util::LoadOp>(loc, valueListPtr);
         
         // Ensure runtime call termination for Hashtable::createIterator (HashMultiMap)
         subop_to_control_flow::RuntimeCallTermination::ensureHashtableCallTermination(
             it.getDefiningOp(), rewriter, loc);
         auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, i8PtrType, valuePtr);
         Block* before = new Block;
         Block* after = new Block;
         whileOp.getBefore().push_back(before);
         whileOp.getAfter().push_back(after);
         mlir::Value beforePtr = before->addArgument(i8PtrType, loc);
         mlir::Value afterPtr = after->addArgument(i8PtrType, loc);
         rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
            mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), beforePtr);
            rewriter.create<mlir::scf::ConditionOp>(loc, cond, beforePtr);
         });
         rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
            Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, valStorageHelper.getStorageType()})), afterPtr);
            Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), castedPtr, 1);
            Value packed = rewriter.create<util::PackOp>(loc, mlir::ValueRange{keyPtr, valuePtr});
            mapping.define(scanRefsOp.getRef(), packed);
            rewriter.replaceTupleStream(scanRefsOp, mapping);
            Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
            mlir::Value next = rewriter.create<util::LoadOp>(loc, nextPtr, mlir::Value());
            rewriter.create<mlir::scf::YieldOp>(loc, next);
         });
      });
      return success();
   }
};

// ============================================================================
// List Scanning Operations
// ============================================================================

class ScanHashMapListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      bool onlyValues = false;
      subop::HashMapType hashmapType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         hashmapType = mlir::dyn_cast_or_null<subop::HashMapType>(lookupRefType.getState());
         onlyValues = true;
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::HashMapEntryRefType>(listType.getT())) {
         hashmapType = entryRefType.getHashMap();
      }

      if (!hashmapType) return mlir::failure();
      auto loc = scanOp.getLoc();
      ColumnMapping mapping;
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), adaptor.getList());
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      auto valPtrType = util::RefType::get(getContext(), mlir::TupleType::get(getContext(), unpackTypes(hashmapType.getValueMembers().getTypes())));
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, scanOp->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, adaptor.getList());
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         if (onlyValues) {
            auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
            mapping.define(scanOp.getElem(), valuePtr);
         } else {
            mapping.define(scanOp.getElem(), kvPtr);
         }
         rewriter.replaceTupleStream(scanOp, mapping);
      });
      return success();
   }
};

class ScanPreAggregationHtListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      bool onlyValues = false;
      subop::PreAggrHtType hashmapType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         hashmapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(lookupRefType.getState());
         onlyValues = true;
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::PreAggrHTEntryRefType>(listType.getT())) {
         hashmapType = entryRefType.getHashMap();
      }

      if (!hashmapType) return mlir::failure();
      auto loc = scanOp.getLoc();
      ColumnMapping mapping;
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), adaptor.getList());
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      auto valPtrType = util::RefType::get(getContext(), mlir::TupleType::get(getContext(), unpackTypes(hashmapType.getValueMembers().getTypes())));
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, scanOp->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, adaptor.getList());
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         if (onlyValues) {
            auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
            mapping.define(scanOp.getElem(), valuePtr);
         } else {
            mapping.define(scanOp.getElem(), kvPtr);
         }
         rewriter.replaceTupleStream(scanOp, mapping);
      });
      return success();
   }
};

class ScanListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT());
      if (!lookupRefType) return mlir::failure();
      auto hashIndexedViewType = mlir::dyn_cast_or_null<subop::HashIndexedViewType>(lookupRefType.getState());
      if (!hashIndexedViewType) return mlir::failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      llvm::SmallVector<mlir::Value> unpacked;
      rewriter.createOrFold<util::UnPackOp>(unpacked, loc, adaptor.getList());
      auto ptr = unpacked[0];
      auto hash = unpacked[1];
      auto initialValid = unpacked[2];
      auto iteratorType = ptr.getType();
      auto referenceType = mlir::cast<subop::ListType>(scanOp.getList().getType()).getT();
      rewriter.create<mlir::scf::IfOp>(
         loc, initialValid, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iteratorType, ptr);
            Block* before = new Block;
            Block* after = new Block;
            whileOp.getBefore().push_back(before);
            whileOp.getAfter().push_back(after);
            rewriter.create<scf::YieldOp>(loc);
            mlir::Value beforePtr = before->addArgument(iteratorType, loc);
            mlir::Value afterPtr = after->addArgument(iteratorType, loc);
            rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
               auto tupleType = mlir::TupleType::get(getContext(), unpackTypes(referenceType.getMembers().getTypes()));
               auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
               Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, rewriter.getIndexType(), tupleType})), beforePtr);
               Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), tupleType), castedPtr, 2);
               if (hashIndexedViewType.getCompareHashForLookup()) {
                  Value hashPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), rewriter.getIndexType()), castedPtr, 1);
                  mlir::Value currHash = rewriter.create<util::LoadOp>(loc, hashPtr, mlir::Value());
                  mlir::Value hashEq = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, currHash, hash);
                  rewriter.create<mlir::scf::IfOp>(
                     loc, hashEq, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
                        mapping.define(scanOp.getElem(), valuePtr);
                        rewriter.replaceTupleStream(scanOp, mapping);
                        builder1.create<mlir::scf::YieldOp>(loc);
                     });
               } else {
                  mapping.define(scanOp.getElem(), valuePtr);
                  rewriter.replaceTupleStream(scanOp, mapping);
               }
               Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
               mlir::Value next = rewriter.create<util::LoadOp>(loc, nextPtr, mlir::Value());
               mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), next);
               rewriter.create<mlir::scf::ConditionOp>(loc, cond, next);
            });
            rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
               rewriter.create<mlir::scf::YieldOp>(loc, afterPtr);
            });
         });

      return success();
   }
};

class ScanExternalHashIndexListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();

      subop::ExternalHashIndexType externalHashIndexType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         if (!(externalHashIndexType = mlir::dyn_cast_or_null<subop::ExternalHashIndexType>(lookupRefType.getState()))) {
            return mlir::failure();
         };
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::ExternalHashIndexEntryRefType>(listType.getT())) {
         externalHashIndexType = entryRefType.getExternalHashIndex();
      } else {
         return mlir::failure();
      }

      auto loc = scanOp->getLoc();
      auto* ctxt = rewriter.getContext();

      // Get correct types
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(externalHashIndexType.getMembers().getTypes()));
      mlir::TypeRange typeRange{tupleType.getTypes()};
      auto i16T = mlir::IntegerType::get(rewriter.getContext(), 16);
      std::vector<mlir::Type> recordBatchTypes{rewriter.getIndexType(), rewriter.getIndexType(), util::RefType::get(i16T), util::RefType::get(rewriter.getI16Type())};
      auto recordBatchInfoRepr = mlir::TupleType::get(ctxt, recordBatchTypes);
      auto convertedListType = typeConverter->convertType(listType);

      // Create while loop to extract all chained values from hash table
      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, convertedListType, adaptor.getList());
      Block* conditionBlock = new Block;
      Block* bodyBlock = new Block;
      whileOp.getBefore().push_back(conditionBlock);
      whileOp.getAfter().push_back(bodyBlock);

      conditionBlock->addArgument(convertedListType, loc);
      bodyBlock->addArgument(convertedListType, loc);
      ColumnMapping mapping;

      // Check if iterator contains another value
      rewriter.atStartOf(conditionBlock, [&](SubOpRewriter& rewriter) {
         mlir::Value list = conditionBlock->getArgument(0);
         // TODO Phase 5: Implement HashIndexIteration::hasNext wrapper
         // mlir::Value cont = rt::HashIndexIteration::hasNext(rewriter, loc)({list})[0];
         mlir::Value cont = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
         rewriter.create<scf::ConditionOp>(loc, cont, ValueRange({list}));
      });

      // Load record batch from iterator
      rewriter.atStartOf(bodyBlock, [&](SubOpRewriter& rewriter) {
         mlir::Value list = bodyBlock->getArgument(0);
         mlir::Value recordBatchPointer;
         rewriter.atStartOf(&scanOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
            recordBatchPointer = rewriter.create<util::AllocaOp>(loc, util::RefType::get(rewriter.getContext(), recordBatchInfoRepr), mlir::Value());
         });
         rt::HashIndexIteration::consumeRecordBatch(rewriter, loc)({list, recordBatchPointer});
         mlir::Value lenRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getIndexType()), recordBatchPointer, 0);
         mlir::Value offsetRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getIndexType()), recordBatchPointer, 1);
         mlir::Value ptrRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(util::RefType::get(rewriter.getI16Type())), recordBatchPointer, 3);
         mlir::Value ptrToColumns = rewriter.create<util::LoadOp>(loc, ptrRef);
         std::vector<mlir::Value> arrays;
         for (size_t i = 0; i < externalHashIndexType.getMembers().getTypes().size(); i++) {
            auto ci = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
            auto array = rewriter.create<util::LoadOp>(loc, ptrToColumns, ci);
            arrays.push_back(array);
         }
         auto arraysVal = rewriter.create<util::PackOp>(loc, arrays);
         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto globalOffset = rewriter.create<util::LoadOp>(loc, offsetRef);
         auto end = rewriter.create<util::LoadOp>(loc, lenRef);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
         rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
            auto withOffset = rewriter.create<mlir::arith::AddIOp>(loc, forOp2.getInductionVar(), globalOffset);
            auto currentRecord = rewriter.create<util::PackOp>(loc, mlir::ValueRange{withOffset, arraysVal});
            mapping.define(scanOp.getElem(), currentRecord);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::scf::YieldOp>(loc, list);
      });

      return success();
   }
};

class ScanMultiMapListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      bool onlyValues = false;
      subop::HashMultiMapType hashMultiMapType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupRefType.getState());
         onlyValues = true;
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(listType.getT())) {
         hashMultiMapType = entryRefType.getHashMultimap();
      }
      if (!hashMultiMapType) return mlir::failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      auto ptr = adaptor.getList();
      EntryStorageHelper keyStorageHelper(scanOp, hashMultiMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(scanOp, hashMultiMapType.getValueMembers(), hashMultiMapType.hasLock(), typeConverter);
      auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
      auto i8PtrPtrType = util::RefType::get(getContext(), i8PtrType);
      Value ptrValid = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), ptr);
      mlir::Value valuePtr = rewriter.create<scf::IfOp>(
                                        loc, ptrValid, [&](OpBuilder& b, Location loc) {
                                           Value valuePtrPtr = rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, ptr, 2);
                                           Value valuePtr = rewriter.create<util::LoadOp>(loc, valuePtrPtr);
                                           b.create<scf::YieldOp>(loc,valuePtr); }, [&](OpBuilder& b, Location loc) {
                                           Value invalidPtr=rewriter.create<util::InvalidRefOp>(loc,i8PtrType);
                                           b.create<scf::YieldOp>(loc, invalidPtr); })
                                .getResult(0);
      Value keyPtr = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ptr, 3);

      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, i8PtrType, valuePtr);
      Block* before = new Block;
      Block* after = new Block;
      whileOp.getBefore().push_back(before);
      whileOp.getAfter().push_back(after);

      mlir::Value beforePtr = before->addArgument(i8PtrType, loc);
      mlir::Value afterPtr = after->addArgument(i8PtrType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), beforePtr);
         rewriter.create<mlir::scf::ConditionOp>(loc, cond, beforePtr);
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
         Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, valStorageHelper.getStorageType()})), afterPtr);
         Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), castedPtr, 1);
         if (onlyValues) {
            mapping.define(scanOp.getElem(), valuePtr);
            rewriter.replaceTupleStream(scanOp, mapping);
         } else {
            Value packed = rewriter.create<util::PackOp>(loc, mlir::ValueRange{keyPtr, valuePtr});
            mapping.define(scanOp.getElem(), packed);
            rewriter.replaceTupleStream(scanOp, mapping);
         }
         Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
         mlir::Value next = rewriter.create<util::LoadOp>(loc, nextPtr, mlir::Value());
         rewriter.create<mlir::scf::YieldOp>(loc, next);
      });
      return success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower