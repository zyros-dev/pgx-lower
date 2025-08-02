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

// Forward declarations for helper functions and classes


// EntryStorageHelper is defined in SubOpToControlFlowUtilities.cpp

////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Thread-Local Merge Operations ////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * MergeThreadLocalResultTable - Merges thread-local result tables
 * 
 * Handles merging of Arrow-compatible result tables from multiple threads.
 * Creates a combine function that merges column builders and updates thread-local storage.
 * 
 * NOTE: In PostgreSQL's single-threaded model, this may need adaptation
 * to handle batch processing or parallel query execution scenarios.
 */
class MergeThreadLocalResultTable : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ResultTableType>(mergeOp.getType())) return mlir::failure();
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "result_table_merge" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftPtr = rewriter.create<util::GenericMemrefCastOp>(loc, typeConverter->convertType(mergeOp.getType()), left);
            auto rightPtr = rewriter.create<util::GenericMemrefCastOp>(loc, typeConverter->convertType(mergeOp.getType()), right);
            auto leftLoaded = rewriter.create<util::LoadOp>(loc, leftPtr);
            auto rightLoaded = rewriter.create<util::LoadOp>(loc, rightPtr);
            auto leftBuilders = rewriter.create<util::UnPackOp>(loc, leftLoaded);
            auto rightBuilders = rewriter.create<util::UnPackOp>(loc, rightLoaded);
            std::vector<mlir::Value> results;
            for (size_t i = 0; i < leftBuilders.getNumResults(); i++) {
               rt::ArrowColumnBuilder::merge(rewriter, loc)({leftBuilders.getResults()[i], rightBuilders.getResults()[i]});
               results.push_back(leftBuilders.getResults()[i]);
            }
            auto packed = rewriter.create<util::PackOp>(loc, results);
            rewriter.create<util::StoreOp>(loc, packed, leftPtr, mlir::Value());
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }

      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));

      mlir::Value merged = rt::ThreadLocal::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};

/**
 * MergeThreadLocalBuffer - Merges thread-local growing buffers
 * 
 * Handles merging of growing buffers from multiple threads using runtime support.
 * Simple merge operation that delegates to the runtime buffer merge function.
 */
class MergeThreadLocalBuffer : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::BufferType>(mergeOp.getType())) return mlir::failure();
      mlir::Value merged = rt::GrowingBuffer::merge(rewriter, mergeOp->getLoc())(adaptor.getThreadLocal())[0];
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};

/**
 * MergeThreadLocalHeap - Merges thread-local heaps
 * 
 * Handles merging of priority heaps from multiple threads using runtime support.
 * Used for top-K operations and priority-based processing.
 */
class MergeThreadLocalHeap : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HeapType>(mergeOp.getType())) return mlir::failure();
      mlir::Value merged = rt::Heap::merge(rewriter, mergeOp->getLoc())(adaptor.getThreadLocal())[0];
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};

/**
 * MergeThreadLocalSimpleState - Merges thread-local simple state objects
 * 
 * Creates a combine function that merges simple state members using provided
 * combine function. Handles nullable types and state member storage.
 * Used for aggregation state and accumulator merging.
 */
class MergeThreadLocalSimpleState : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::SimpleStateType>(mergeOp.getType())) return mlir::failure();
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();
      auto simpleStateType = mlir::cast<subop::SimpleStateType>(mergeOp.getType());
      EntryStorageHelper storageHelper(mergeOp, simpleStateType.getMembers(), simpleStateType.hasLock(), typeConverter);

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "simple_state__combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value dest = left;
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftValues = storageHelper.getValueMap(left, rewriter, loc);
            auto rightValues = storageHelper.getValueMap(right, rewriter, loc);
            std::vector<mlir::Value> args;
            for (const auto& pair : leftValues) { args.push_back(pair.second); }
            for (const auto& pair : rightValues) { args.push_back(pair.second); }
            for (size_t i = 0; i < args.size(); i++) {
               auto expectedType = mergeOp.getCombineFn().front().getArgument(i).getType();
               if (args[i].getType() != expectedType) {
                  args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
               }
            }
            Block* sortLambda = &mergeOp.getCombineFn().front();
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
               storageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
            });
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }

      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      mlir::Value merged = rt::SimpleState::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};

/**
 * MergeThreadLocalHashMap - Merges thread-local hash maps
 * 
 * Creates both equality and combine functions for merging hash map entries.
 * Handles key comparison and value merging for hash-based aggregation.
 * Used for group-by operations and hash-based joins.
 */
class MergeThreadLocalHashMap : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMapType>(mergeOp.getType())) return mlir::failure();
      auto hashMapType = mlir::cast<subop::HashMapType>(mergeOp.getType());
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      mlir::func::FuncOp eqFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();
      EntryStorageHelper keyStorageHelper(mergeOp, hashMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(mergeOp, hashMapType.getValueMembers(), hashMapType.hasLock(), typeConverter);

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         eqFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "hashmap_eq_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "hashmap_combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            if (!mergeOp.getCombineFn().empty()) {
               auto kvType = getHtKVType(hashMapType, *typeConverter);
               auto kvPtrType = util::RefType::get(context, kvType);
               left = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, left);
               right = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, right);

               left = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), left, 1);
               right = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), right, 1);
               Value dest = left;
               auto leftValues = valStorageHelper.getValueMap(left, rewriter, loc);
               auto rightValues = valStorageHelper.getValueMap(right, rewriter, loc);
               std::vector<mlir::Value> args;
               for (const auto& pair : leftValues) { args.push_back(pair.second); }
               for (const auto& pair : rightValues) { args.push_back(pair.second); }
               for (size_t i = 0; i < args.size(); i++) {
                  auto expectedType = mergeOp.getCombineFn().front().getArgument(i).getType();
                  if (args[i].getType() != expectedType) {
                     args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
                  }
               }
               Block* sortLambda = &mergeOp.getCombineFn().front();
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
                  valStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
               });
            }
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         eqFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftKeys = keyStorageHelper.getValueMap(left, rewriter, loc);
            auto rightKeys = keyStorageHelper.getValueMap(right, rewriter, loc);
            std::vector<mlir::Value> args;
            for (const auto& pair : leftKeys) { args.push_back(pair.second); }
            for (const auto& pair : rightKeys) { args.push_back(pair.second); }
            auto res = inlineBlock(&mergeOp.getEqFn().front(), rewriter, args)[0];
            rewriter.create<mlir::func::ReturnOp>(loc, res);
         });
      }

      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      Value eqFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, eqFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(eqFn.getSymName())));
      mlir::Value merged = rt::Hashtable::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), eqFnPtr, combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};

/**
 * MergePreAggrHashMap - Merges thread-local pre-aggregation hash maps
 * 
 * Similar to MergeThreadLocalHashMap but specialized for pre-aggregation scenarios.
 * Creates optimized combine and equality functions for pre-aggregation hash tables.
 * Used for optimistic hash-based aggregation with conflict resolution.
 */
class MergePreAggrHashMap : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtType>(mergeOp.getType())) return mlir::failure();
      auto hashMapType = mlir::cast<subop::PreAggrHtType>(mergeOp.getType());
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      mlir::func::FuncOp eqFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();
      EntryStorageHelper keyStorageHelper(mergeOp, hashMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(mergeOp, hashMapType.getValueMembers(), hashMapType.hasLock(), typeConverter);

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         eqFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "optimistic_ht_eq_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "optimistic_ht_combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            if (!mergeOp.getCombineFn().empty()) {
               auto kvType = getHtKVType(hashMapType, *typeConverter);
               auto kvPtrType = util::RefType::get(context, kvType);
               left = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, left);
               right = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, right);

               left = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), left, 1);
               right = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), right, 1);
               Value dest = left;
               auto leftValues = valStorageHelper.getValueMap(left, rewriter, loc);
               auto rightValues = valStorageHelper.getValueMap(right, rewriter, loc);
               std::vector<mlir::Value> args;
               for (const auto& pair : leftValues) { args.push_back(pair.second); }
               for (const auto& pair : rightValues) { args.push_back(pair.second); }
               for (size_t i = 0; i < args.size(); i++) {
                  auto expectedType = mergeOp.getCombineFn().front().getArgument(i).getType();
                  if (args[i].getType() != expectedType) {
                     args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
                  }
               }
               Block* sortLambda = &mergeOp.getCombineFn().front();
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
                  valStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
               });
            }
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         eqFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftKeys = keyStorageHelper.getValueMap(left, rewriter, loc);
            auto rightKeys = keyStorageHelper.getValueMap(right, rewriter, loc);
            std::vector<mlir::Value> args;
            for (const auto& pair : leftKeys) { args.push_back(pair.second); }
            for (const auto& pair : rightKeys) { args.push_back(pair.second); }
            auto res = inlineBlock(&mergeOp.getEqFn().front(), rewriter, args)[0];
            rewriter.create<mlir::func::ReturnOp>(loc, res);
         });
      }
      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      Value eqFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, eqFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(eqFn.getSymName())));
      mlir::Value merged = rt::PreAggregationHashtable::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), eqFnPtr, combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower