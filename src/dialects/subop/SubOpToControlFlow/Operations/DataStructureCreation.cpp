#include "../Headers/SubOpToControlFlowCommon.h"
#include "../Headers/SubOpToControlFlowPatterns.h"
#include "../Headers/SubOpToControlFlowRewriter.h"
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

// Forward declarations of helper functions
class EntryStorageHelper;
static mlir::TupleType getHtEntryType(subop::HashMapType t, mlir::TypeConverter& converter);
static mlir::TupleType getHtEntryType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter);
static mlir::TupleType getHashMultiMapEntryType(subop::HashMultiMapType t, mlir::TypeConverter& converter);
static mlir::TupleType getHashMultiMapValueType(subop::HashMultiMapType t, mlir::TypeConverter& converter);

/**
 * CreateThreadLocalLowering - Lowers thread-local storage creation operations.
 * 
 * This lowering pattern creates thread-local storage by:
 * 1. Creating an initialization function that sets up the storage
 * 2. Allocating raw memory for parameter storage
 * 3. Setting up function pointers for thread-local access
 * 4. Integrating with PostgreSQL memory contexts
 */
class CreateThreadLocalLowering : public SubOpConversionPattern<subop::CreateThreadLocalOp> {
   public:
   using SubOpConversionPattern<subop::CreateThreadLocalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateThreadLocalOp createThreadLocal, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      ModuleOp parentModule = createThreadLocal->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;
      auto loc = createThreadLocal->getLoc();
      auto i8PtrType = util::RefType::get(rewriter.getI8Type());
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "thread_local_init" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({i8PtrType}), TypeRange(i8PtrType)));
      });
      auto* funcBody = new Block;
      mlir::Value funcArg = funcBody->addArgument(i8PtrType, loc);
      funcOp.getBody().push_back(funcBody);
      mlir::Block* newBlock = new Block;
      mlir::OpBuilder builder(rewriter.getContext());
      std::vector<mlir::Operation*> toInsert;
      builder.setInsertionPointToStart(newBlock);
      mlir::Value argRef = newBlock->addArgument(util::RefType::get(rewriter.getI8Type()), createThreadLocal.getLoc());
      argRef = builder.create<util::GenericMemrefCastOp>(createThreadLocal->getLoc(), util::RefType::get(rewriter.getContext(), util::RefType::get(rewriter.getContext(), rewriter.getI8Type())), argRef);
      for (auto& op : createThreadLocal.getInitFn().front().getOperations()) {
         toInsert.push_back(&op);
      }
      for (auto* op : toInsert) {
         op->remove();
         builder.insert(op);
      }
      std::vector<mlir::Value> toStore;
      std::vector<mlir::Operation*> toDelete;
      newBlock->walk([&](tuples::GetParamVal op) {
         builder.setInsertionPointAfter(op);
         auto idx = toStore.size();
         toStore.push_back(op.getParam());
         mlir::Value rawPtr = builder.create<util::LoadOp>(createThreadLocal->getLoc(), argRef, builder.create<mlir::arith::ConstantIndexOp>(loc, idx));
         mlir::Value ptr = builder.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getContext(), op.getParam().getType()), rawPtr);
         mlir::Value value = builder.create<util::LoadOp>(loc, ptr);
         op.replaceAllUsesWith(value);
         toDelete.push_back(op);
      });
      for (auto* op : toDelete) {
         op->erase();
      }
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(newBlock, mlir::ValueRange{funcArg}, [&](tuples::ReturnOpAdaptor adaptor) {
            mlir::Value unrealized = rewriter.create<mlir::UnrealizedConversionCastOp>(createThreadLocal->getLoc(), createThreadLocal.getType().getWrapped(), adaptor.getResults()[0]).getOutputs()[0];
            mlir::Value casted = rewriter.create<util::GenericMemrefCastOp>(createThreadLocal->getLoc(), util::RefType::get(rewriter.getI8Type()), unrealized);
            rewriter.create<mlir::func::ReturnOp>(loc, casted);
         });
         delete newBlock;
      });
      auto ptrSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), i8PtrType);
      auto numPtrs = rewriter.create<mlir::arith::ConstantIndexOp>(loc, toStore.size());
      auto bytes = rewriter.create<mlir::arith::MulIOp>(loc, ptrSize, numPtrs);

      Value arg = rt::ExecutionContext::allocStateRaw(rewriter, loc)({bytes})[0];
      arg = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getContext(), i8PtrType), arg);
      for (size_t i = 0; i < toStore.size(); i++) {
         Value storeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), toStore[i].getType());
         Value valPtrOrig = rt::ExecutionContext::allocStateRaw(rewriter, loc)({storeSize})[0];
         Value valPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(toStore[i].getType()), valPtrOrig);
         rewriter.create<util::StoreOp>(loc, toStore[i], valPtr, mlir::Value());
         rewriter.create<util::StoreOp>(loc, valPtrOrig, arg, rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
      }
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      rewriter.replaceOp(createThreadLocal, rt::ThreadLocal::create(rewriter, loc)({functionPointer, arg}));
      return mlir::success();
   }
};

/**
 * CreateBufferLowering - Lowers buffer creation operations.
 * 
 * Creates growing buffers with initial capacity and allocator support.
 * Supports group allocators for memory management optimization.
 */
class CreateBufferLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(createOp.getType());
      if (!bufferType) return failure();
      auto loc = createOp->getLoc();
      EntryStorageHelper storageHelper(createOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, createOp->hasAttr("initial_capacity") ? mlir::cast<mlir::IntegerAttr>(createOp->getAttr("initial_capacity")).getInt() : 1024);
      auto elementType = storageHelper.getStorageType();
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      mlir::Value allocator;
      rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
         if (createOp->hasAttrOfType<mlir::IntegerAttr>("group")) {
            Value groupId = rewriter.create<arith::ConstantIndexOp>(loc, mlir::cast<mlir::IntegerAttr>(createOp->getAttr("group")).getInt());
            allocator = rt::GrowingBufferAllocator::getGroupAllocator(rewriter, loc)({groupId})[0];
         } else {
            allocator = rt::GrowingBufferAllocator::getDefaultAllocator(rewriter, loc)({})[0];
         }
      });
      mlir::Value vector = rt::GrowingBuffer::create(rewriter, loc)({allocator, typeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, vector);
      return mlir::success();
   }
};

/**
 * CreateHashMapLowering - Lowers hash map creation operations.
 * 
 * Creates hash tables with specified entry types and initial capacity.
 */
class CreateHashMapLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMapType>(createOp.getType())) return failure();
      auto t = mlir::cast<subop::HashMapType>(createOp.getType());

      auto typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHtEntryType(t, *typeConverter));
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(createOp->getLoc(), 4);
      auto ptr = rt::Hashtable::create(rewriter, createOp->getLoc())({typeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};

/**
 * CreateHashMultiMapLowering - Lowers hash multi-map creation operations.
 * 
 * Creates hash multi-maps with entry and value type sizes.
 */
class CreateHashMultiMapLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMultiMapType>(createOp.getType())) return failure();
      auto t = mlir::cast<subop::HashMultiMapType>(createOp.getType());

      auto entryTypeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHashMultiMapEntryType(t, *typeConverter));
      auto valueTypeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHashMultiMapValueType(t, *typeConverter));
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(createOp->getLoc(), 4);
      auto ptr = rt::HashMultiMap::create(rewriter, createOp->getLoc())({entryTypeSize, valueTypeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};

/**
 * CreateOpenHtFragmentLowering - Lowers open hash table fragment creation.
 * 
 * Creates pre-aggregation hash table fragments with lock support.
 */
class CreateOpenHtFragmentLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtFragmentType>(createOp.getType())) return failure();
      auto t = mlir::cast<subop::PreAggrHtFragmentType>(createOp.getType());

      auto typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHtEntryType(t, *typeConverter));
      auto withLocks = rewriter.create<mlir::arith::ConstantIntOp>(createOp->getLoc(), t.getWithLock(), rewriter.getI1Type());
      auto ptr = rt::PreAggregationHashtableFragment::create(rewriter, createOp->getLoc())({typeSize, withLocks})[0];
      rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(createOp, typeConverter->convertType(t), ptr);
      return mlir::success();
   }
};

/**
 * CreateArrayLowering - Lowers array creation operations.
 * 
 * Creates zeroed buffers with specified element count and storage layout.
 */
class CreateArrayLowering : public SubOpConversionPattern<subop::CreateArrayOp> {
   public:
   using SubOpConversionPattern<subop::CreateArrayOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateArrayOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto arrayType = createOp.getType();
      auto loc = createOp->getLoc();
      EntryStorageHelper storageHelper(createOp, arrayType.getMembers(), arrayType.hasLock(), typeConverter);

      Value tpl = rewriter.create<util::LoadOp>(loc, adaptor.getNumElements());
      Value numElements = rewriter.create<util::UnPackOp>(loc, tpl).getResults()[0];
      auto elementType = storageHelper.getStorageType();
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      auto numBytes = rewriter.create<mlir::arith::MulIOp>(loc, typeSize, numElements);
      mlir::Value vector = rt::Buffer::createZeroed(rewriter, loc)({numBytes})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(createOp, typeConverter->convertType(createOp.getType()), vector);
      return mlir::success();
   }
};

/**
 * CreateSegmentTreeViewLowering - Lowers segment tree view creation.
 * 
 * Creates segment tree views with initial and combine functions for aggregation.
 * Generates initialization and combination function pointers for runtime use.
 */
class CreateSegmentTreeViewLowering : public SubOpConversionPattern<subop::CreateSegmentTreeView> {
   public:
   using SubOpConversionPattern<subop::CreateSegmentTreeView>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateSegmentTreeView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      auto loc = createOp->getLoc();
      auto continuousType = createOp.getSource().getType();

      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));

      ModuleOp parentModule = createOp->getParentOfType<ModuleOp>();
      EntryStorageHelper sourceStorageHelper(createOp, continuousType.getMembers(), continuousType.hasLock(), typeConverter);
      EntryStorageHelper viewStorageHelper(createOp, createOp.getType().getValueMembers(), continuousType.hasLock(), typeConverter);
      mlir::TupleType sourceElementType = sourceStorageHelper.getStorageType();
      mlir::TupleType viewElementType = viewStorageHelper.getStorageType();

      mlir::func::FuncOp initialFn;
      mlir::func::FuncOp combineFn;
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         initialFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "segment_tree_initial_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "segment_tree_combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
         initialFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            Value dest = funcBody->getArgument(0);
            Value src = funcBody->getArgument(1);
            auto sourceValues = sourceStorageHelper.getValueMap(src, rewriter, loc);
            std::vector<mlir::Value> args;
            for (auto relevantMember : createOp.getRelevantMembers()) {
               args.push_back(sourceValues.get(mlir::cast<mlir::StringAttr>(relevantMember).str()));
            }
            Block* sortLambda = &createOp.getInitialFn().front();
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
               viewStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
            });
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }
      {
         auto* funcBody = new Block;
         Value dest = funcBody->addArgument(ptrType, loc);
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftValues = viewStorageHelper.getValueMap(left, rewriter, loc);
            auto rightValues = viewStorageHelper.getValueMap(right, rewriter, loc);
            std::vector<mlir::Value> args;
            args.insert(args.end(), leftValues.begin(), leftValues.end());
            args.insert(args.end(), rightValues.begin(), rightValues.end());
            Block* sortLambda = &createOp.getCombineFn().front();
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
               viewStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
            });
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }

      Value initialFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, initialFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(initialFn.getSymName())));
      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      Value sourceEntryTypeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), sourceElementType);
      Value stateTypeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), viewElementType);
      mlir::Value res = rt::SegmentTreeView::build(rewriter, loc)({adaptor.getSource(), sourceEntryTypeSize, initialFnPtr, combineFnPtr, stateTypeSize})[0];
      rewriter.replaceOp(createOp, res);
      return mlir::success();
   }
};

/**
 * CreateHeapLowering - Lowers heap creation operations.
 * 
 * Creates heaps with comparison functions for priority queue operations.
 * Generates comparison function pointer for runtime heap management.
 */
class CreateHeapLowering : public SubOpConversionPattern<subop::CreateHeapOp> {
   public:
   using SubOpConversionPattern<subop::CreateHeapOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateHeapOp heapOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      auto heapType = heapOp.getType();
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      EntryStorageHelper storageHelper(heapOp, heapType.getMembers(), heapType.hasLock(), typeConverter);
      ModuleOp parentModule = heapOp->getParentOfType<ModuleOp>();
      mlir::TupleType elementType = storageHelper.getStorageType();
      auto loc = heapOp.getLoc();
      mlir::func::FuncOp funcOp;
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "heap_compare" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
      });
      auto* funcBody = new Block;
      Value left = funcBody->addArgument(ptrType, loc);
      Value right = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto leftVals = storageHelper.getValueMap(left, rewriter, loc, heapOp.getSortBy());
         auto rightVals = storageHelper.getValueMap(right, rewriter, loc, heapOp.getSortBy());
         std::vector<mlir::Value> args;
         args.insert(args.end(), leftVals.begin(), leftVals.end());
         args.insert(args.end(), rightVals.begin(), rightVals.end());
         Block* sortLambda = &heapOp.getRegion().front();
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
            rewriter.create<mlir::func::ReturnOp>(loc, adaptor.getResults());
         });
      });
      Value typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      Value maxElements = rewriter.create<mlir::arith::ConstantIndexOp>(loc, heapType.getMaxElements());
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      auto heap = rt::Heap::create(rewriter, loc)({maxElements, typeSize, functionPointer})[0];
      rewriter.replaceOp(heapOp, heap);
      return mlir::success();
   }
};

/**
 * CreateHashIndexedViewLowering - Lowers hash indexed view creation.
 * 
 * Creates hash indexed views for efficient hash-based lookups.
 * Validates that link member is first and hash member is second.
 */
class CreateHashIndexedViewLowering : public SubOpConversionPattern<subop::CreateHashIndexedView> {
   using SubOpConversionPattern<subop::CreateHashIndexedView>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateHashIndexedView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast<subop::BufferType>(createOp.getSource().getType());
      if (!bufferType) return failure();
      auto linkIsFirst = mlir::cast<mlir::StringAttr>(bufferType.getMembers().getNames()[0]).str() == createOp.getLinkMember();
      auto hashIsSecond = mlir::cast<mlir::StringAttr>(bufferType.getMembers().getNames()[1]).str() == createOp.getHashMember();
      if (!linkIsFirst || !hashIsSecond) return failure();
      auto htView = rt::HashIndexedView::build(rewriter, createOp->getLoc())({adaptor.getSource()})[0];
      rewriter.replaceOp(createOp, htView);
      return success();
   }
};

/**
 * CreateContinuousViewLowering - Lowers continuous view creation.
 * 
 * Creates continuous views from buffers, arrays, or sorted views.
 * Handles type conversion and buffer casting as needed.
 */
class CreateContinuousViewLowering : public SubOpConversionPattern<subop::CreateContinuousView> {
   using SubOpConversionPattern<subop::CreateContinuousView>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateContinuousView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (mlir::isa<subop::ArrayType>(createOp.getSource().getType())) {
         // For now: every array is equivalent to continuous view
         rewriter.replaceOp(createOp, adaptor.getSource());
         return success();
      }
      if (mlir::isa<subop::SortedViewType>(createOp.getSource().getType())) {
         // For now: every sorted view is equivalent to continuous view
         rewriter.replaceOp(createOp, adaptor.getSource());
         return success();
      }
      auto bufferType = mlir::dyn_cast<subop::BufferType>(createOp.getSource().getType());
      if (!bufferType) return failure();
      auto genericBuffer = rt::GrowingBuffer::asContinuous(rewriter, createOp->getLoc())({adaptor.getSource()})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(createOp, typeConverter->convertType(createOp.getType()), genericBuffer);
      return success();
   }
};

/**
 * GetExternalHashIndexLowering - Lowers external hash index access.
 * 
 * Creates access to external hash indices using relation helper.
 */
class GetExternalHashIndexLowering : public SubOpConversionPattern<subop::GetExternalOp> {
   public:
   using SubOpConversionPattern<subop::GetExternalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GetExternalOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ExternalHashIndexType>(op.getType())) return failure();
      mlir::Value description = rewriter.create<util::CreateConstVarLen>(op->getLoc(), util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());

      rewriter.replaceOp(op, rt::RelationHelper::accessHashIndex(rewriter, op->getLoc())({description})[0]);
      return mlir::success();
   }
};

/**
 * CreateSimpleStateLowering - Lowers simple state creation operations.
 * 
 * Creates simple state objects either on heap or stack with optional initialization.
 * Integrates with PostgreSQL memory contexts for proper memory management.
 */
class CreateSimpleStateLowering : public SubOpConversionPattern<subop::CreateSimpleStateOp> {
   public:
   using SubOpConversionPattern<subop::CreateSimpleStateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateSimpleStateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto simpleStateType = mlir::dyn_cast_or_null<subop::SimpleStateType>(createOp.getType());
      if (!simpleStateType) return failure();

      mlir::Value ref;
      if (createOp->hasAttr("allocateOnHeap")) {
         auto loweredType = mlir::cast<util::RefType>(typeConverter->convertType(createOp.getType()));
         mlir::Value typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), loweredType.getElementType());
         ref = rt::SimpleState::create(rewriter, createOp->getLoc())(mlir::ValueRange{typeSize})[0];
         ref = rewriter.create<util::GenericMemrefCastOp>(createOp->getLoc(), loweredType, ref);

      } else {
         rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
            ref = rewriter.create<util::AllocaOp>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), mlir::Value());
         });
      }
      if (!createOp.getInitFn().empty()) {
         EntryStorageHelper storageHelper(createOp, simpleStateType.getMembers(), simpleStateType.hasLock(), typeConverter);
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&createOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor returnOpAdaptor) {
            storageHelper.storeOrderedValues(ref, returnOpAdaptor.getResults(), rewriter, createOp->getLoc());
         });
      }
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower