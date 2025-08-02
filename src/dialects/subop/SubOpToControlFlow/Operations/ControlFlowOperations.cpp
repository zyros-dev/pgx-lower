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

// Helper template function for repeating values
template <class T>
static std::vector<T> repeat(T val, size_t times) {
   std::vector<T> res{};
   for (auto i = 0ul; i < times; i++) res.push_back(val);
   return res;
}

//===----------------------------------------------------------------------===//
// MapLowering - Critical for expression evaluation (Tests 9-15)
//===----------------------------------------------------------------------===//

class MapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MapOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MapOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MapOp mapOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto args = mapping.resolve(mapOp, mapOp.getInputCols());
      std::vector<Value> res;

      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&mapOp.getFn().front(), args, [&](tuples::ReturnOpAdaptor adaptor) {
         res.insert(res.end(), adaptor.getResults().begin(), adaptor.getResults().end());
      });
      for (auto& r : res) {
         r = rewriter.getMapped(r);
      }
      mapping.define(mapOp.getComputedCols(), res);

      rewriter.replaceTupleStream(mapOp, mapping);
      return success();
   }
};

//===----------------------------------------------------------------------===//
// InFlightLowering - In-flight operations
//===----------------------------------------------------------------------===//

class InFlightLowering : public SubOpConversionPattern<subop::InFlightOp> {
   public:
   using SubOpConversionPattern<subop::InFlightOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::InFlightOp inFlightOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      ColumnMapping mapping(inFlightOp);
      rewriter.replaceTupleStream(inFlightOp, mapping);
      return success();
   }
};

//===----------------------------------------------------------------------===//
// FilterLowering - Filter operations with conditional execution
//===----------------------------------------------------------------------===//

class FilterLowering : public SubOpTupleStreamConsumerConversionPattern<subop::FilterOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::FilterOp>::SubOpTupleStreamConsumerConversionPattern;
   
   LogicalResult matchAndRewrite(subop::FilterOp filterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto providedVals = mapping.resolve(filterOp, filterOp.getConditions());
      mlir::Value cond;
      if (providedVals.size() == 1) {
         cond = providedVals[0];
      } else {
         cond = rewriter.create<db::AndOp>(filterOp.getLoc(), mlir::ValueRange(providedVals), llvm::ArrayRef<mlir::NamedAttribute>{});
      }
      if (!cond.getType().isInteger(1)) {
         cond = rewriter.create<db::DeriveTruth>(filterOp.getLoc(), cond);
      }
      if (filterOp.getFilterSemantic() == subop::FilterSemantic::none_true) {
         cond = rewriter.create<db::NotOp>(filterOp->getLoc(), cond);
      }
      auto ifOp = rewriter.create<mlir::scf::IfOp>(filterOp->getLoc(), mlir::TypeRange{}, cond);
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, filterOp->getLoc());
      rewriter.atStartOf(ifOp.thenBlock(), [&](SubOpRewriter& rewriter) {
         rewriter.replaceTupleStream(filterOp, mapping);
      });
      return success();
   }
};

//===----------------------------------------------------------------------===//
// RenameLowering - Column renaming operations
//===----------------------------------------------------------------------===//

class RenameLowering : public SubOpTupleStreamConsumerConversionPattern<subop::RenamingOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::RenamingOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::RenamingOp renamingOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      for (mlir::Attribute attr : renamingOp.getColumns()) {
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         mlir::Attribute from = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[0];
         auto relationRefAttr = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(from);
         mapping.define(relationDefAttr, mapping.resolve(renamingOp, relationRefAttr));
      }
      rewriter.replaceTupleStream(renamingOp, mapping);
      return success();
   }
};

//===----------------------------------------------------------------------===//
// NestedMapLowering - Nested map operations for complex execution patterns
//===----------------------------------------------------------------------===//

class NestedMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::NestedMapOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::NestedMapOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::NestedMapOp nestedMapOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      for (auto [p, a] : llvm::zip(nestedMapOp.getParameters(), nestedMapOp.getRegion().front().getArguments().drop_front())) {
         rewriter.map(a, mapping.resolve(nestedMapOp, mlir::cast<tuples::ColumnRefAttr>(p)));
      }
      auto nestedExecutionGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(&nestedMapOp.getRegion().front().front());
      if (!nestedExecutionGroup) {
         nestedMapOp.emitError("NestedMapOp should have a NestedExecutionGroupOp as the first operation in the region");
         return failure();
      }
      auto returnOp = mlir::cast<tuples::ReturnOp>(nestedMapOp.getRegion().front().getTerminator());
      if (!returnOp.getResults().empty()) {
         nestedMapOp.emitError("NestedMapOp should not return any values for the lowering");
         return failure();
      }
      mlir::IRMapping outerMapping;
      for (auto [i, b] : llvm::zip(nestedExecutionGroup.getInputs(), nestedExecutionGroup.getRegion().front().getArguments())) {
         outerMapping.map(b, rewriter.getMapped(i));
      }
      for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
         if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
            auto guard = rewriter.nest(outerMapping, step);
            for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
               mlir::Value input = outerMapping.lookup(param);
               rewriter.map(arg, input);
            }
            mlir::IRMapping cloneMapping;
            std::vector<mlir::Operation*> ops;
            for (auto& op : step.getSubOps().front()) {
               if (&op == step.getSubOps().front().getTerminator())
                  break;
               ops.push_back(&op);
            }
            for (auto* op : ops) {
               op->remove();
               rewriter.insertAndRewrite(op);
            }
            auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
            for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
               auto mapped = rewriter.getMapped(i);
               outerMapping.map(o, mapped);
            }
         }
      }
      rewriter.eraseOp(nestedMapOp);
      return success();
   }
};

//===----------------------------------------------------------------------===//
// LoopLowering - Loop operations with SCF integration
//===----------------------------------------------------------------------===//

class LoopLowering : public SubOpConversionPattern<subop::LoopOp> {
   public:
   using SubOpConversionPattern<subop::LoopOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::LoopOp loopOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto loc = loopOp->getLoc();
      auto* b = loopOp.getBody();
      auto* terminator = b->getTerminator();
      auto continueOp = mlir::cast<subop::LoopContinueOp>(terminator);
      mlir::Value trueValue = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, rewriter.getI1Type());
      std::vector<mlir::Type> iterTypes;
      std::vector<mlir::Value> iterArgs;

      iterTypes.push_back(rewriter.getI1Type());
      for (auto argumentType : loopOp.getBody()->getArgumentTypes()) {
         iterTypes.push_back(typeConverter->convertType(argumentType));
      }
      iterArgs.push_back(trueValue);
      iterArgs.insert(iterArgs.end(), adaptor.getArgs().begin(), adaptor.getArgs().end());
      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iterTypes, iterArgs);
      auto* before = new Block;
      before->addArguments(iterTypes, repeat(loc, iterTypes.size()));
      whileOp.getBefore().push_back(before);

      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         rewriter.create<mlir::scf::ConditionOp>(loc, before->getArgument(0), before->getArguments());
      });

      auto nestedExecutionGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(&loopOp.getBody()->front());
      if (!nestedExecutionGroup) {
         loopOp.emitError("LoopOp should have a NestedExecutionGroupOp as the first operation in the region");
         return failure();
      }
      auto* after = new Block;
      after->addArguments(iterTypes, repeat(loc, iterTypes.size()));
      whileOp.getAfter().push_back(after);
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         std::vector<mlir::Value> args;
         for (size_t i = 0; i < loopOp.getBody()->getNumArguments(); i++) {
            mlir::Value whileArg = after->getArguments()[i + 1];
            rewriter.map(loopOp.getBody()->getArgument(i), whileArg);
         }
         mlir::IRMapping nestedGroupResultMapping;

         mlir::IRMapping outerMapping;
         for (auto [i, b] : llvm::zip(nestedExecutionGroup.getInputs(), nestedExecutionGroup.getRegion().front().getArguments())) {
            outerMapping.map(b, rewriter.getMapped(i));
         }
         for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
            if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
               auto guard = rewriter.nest(outerMapping, step);
               for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
                  mlir::Value input = outerMapping.lookup(param);
                  rewriter.map(arg, input);
               }
               mlir::IRMapping cloneMapping;
               std::vector<mlir::Operation*> ops;
               for (auto& op : step.getSubOps().front()) {
                  if (&op == step.getSubOps().front().getTerminator())
                     break;
                  ops.push_back(&op);
               }
               for (auto* op : ops) {
                  rewriter.operator mlir::OpBuilder&().setInsertionPointToEnd(after);
                  op->remove();
                  rewriter.insertAndRewrite(op);
               }
               auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
               for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
                  auto mapped = rewriter.getMapped(i);
                  outerMapping.map(o, mapped);
               }
            } else if (auto returnOp = mlir::dyn_cast_or_null<subop::NestedExecutionGroupReturnOp>(&op)) {
               for (auto [i, o] : llvm::zip(returnOp.getInputs(), nestedExecutionGroup.getResults())) {
                  nestedGroupResultMapping.map(o, outerMapping.lookup(i));
               }
            }
         }
         rewriter.operator mlir::OpBuilder&().setInsertionPointToEnd(after);
         std::vector<mlir::Value> res;
         auto simpleStateType = mlir::cast<subop::SimpleStateType>(continueOp.getOperandTypes()[0]);
         EntryStorageHelper storageHelper(loopOp, simpleStateType.getMembers(), simpleStateType.hasLock(), typeConverter);
         auto shouldContinueBool = storageHelper.getValueMap(nestedGroupResultMapping.lookup(continueOp.getOperand(0)), rewriter, loc).get(continueOp.getCondMember().str());
         res.push_back(shouldContinueBool);
         for (auto operand : continueOp->getOperands().drop_front()) {
            res.push_back(nestedGroupResultMapping.lookup(operand));
         }
         rewriter.create<mlir::scf::YieldOp>(loc, res);
      });
      rewriter.replaceOp(loopOp, whileOp.getResults().drop_front());
      return success();
   }
};

//===----------------------------------------------------------------------===//
// NestedExecutionGroupLowering - Nested execution groups
//===----------------------------------------------------------------------===//

class NestedExecutionGroupLowering : public SubOpConversionPattern<subop::NestedExecutionGroupOp> {
   public:
   using SubOpConversionPattern<subop::NestedExecutionGroupOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::NestedExecutionGroupOp nestedExecutionGroup, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto loc = nestedExecutionGroup->getLoc();
      auto dummyOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getI1Type());
      rewriter.operator mlir::OpBuilder&().setInsertionPoint(dummyOp);
      mlir::IRMapping nestedGroupResultMapping;

      mlir::IRMapping outerMapping;
      for (auto [i, b] : llvm::zip(adaptor.getInputs(), nestedExecutionGroup.getRegion().front().getArguments())) {
         outerMapping.map(b, i);
      }
      std::vector<mlir::Value> toReplaceWith;
      for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
         if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
            auto guard = rewriter.nest(outerMapping, step);
            for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
               mlir::Value input = outerMapping.lookup(param);
               rewriter.map(arg, input);
            }
            mlir::IRMapping cloneMapping;
            std::vector<mlir::Operation*> ops;
            for (auto& op : step.getSubOps().front()) {
               if (&op == step.getSubOps().front().getTerminator())
                  break;
               ops.push_back(&op);
            }
            for (auto* op : ops) {
               rewriter.operator mlir::OpBuilder&().setInsertionPoint(dummyOp);
               op->remove();
               rewriter.insertAndRewrite(op);
            }
            auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
            for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
               auto mapped = rewriter.getMapped(i);
               outerMapping.map(o, mapped);
            }
         } else if (auto returnOp = mlir::dyn_cast_or_null<subop::NestedExecutionGroupReturnOp>(&op)) {
            for (auto [i, o] : llvm::zip(returnOp.getInputs(), nestedExecutionGroup.getResults())) {
               toReplaceWith.push_back(outerMapping.lookup(i));
            }
         }
      }
      rewriter.replaceOp(nestedExecutionGroup, toReplaceWith);
      return success();
   }
};

//===----------------------------------------------------------------------===//
// SetTrackedCountLowering - Tracked count operations
//===----------------------------------------------------------------------===//

class SetTrackedCountLowering : public SubOpConversionPattern<subop::SetTrackedCountOp> {
   public:
   using SubOpConversionPattern<subop::SetTrackedCountOp>::SubOpConversionPattern;
   
   LogicalResult matchAndRewrite(subop::SetTrackedCountOp setTrackedCountOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto loc = setTrackedCountOp->getLoc();
      // Get resultId
      mlir::Value resultId = rewriter.create<mlir::arith::ConstantIntOp>(loc, setTrackedCountOp.getResultId(), mlir::IntegerType::get(rewriter.getContext(), 32));

      // Get tupleCount
      Value loadedTuple = rewriter.create<util::LoadOp>(loc, adaptor.getTupleCount());
      Value tupleCount = rewriter.create<util::UnPackOp>(loc, loadedTuple).getResults()[0];

      rt::ExecutionContext::setTupleCount(rewriter, loc)({resultId, tupleCount});
      rewriter.eraseOp(setTrackedCountOp);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// LockLowering - Locking operations for thread safety
//===----------------------------------------------------------------------===//

class LockLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LockOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LockOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::LockOp lockOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = lockOp.getRef().getColumn().type;
      auto entryRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(refType);
      if (!entryRefType) return failure();
      auto hashMapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(entryRefType.getState());
      if (!hashMapType || !hashMapType.getWithLock()) return failure();
      assert(hashMapType.hasLock());
      auto storageHelper = EntryStorageHelper(lockOp, hashMapType.getValueMembers(), hashMapType.hasLock(), typeConverter);
      auto ref = mapping.resolve(lockOp, lockOp.getRef());
      auto lockPtr = storageHelper.getLockPointer(ref, rewriter, lockOp->getLoc());
      rt::EntryLock::lock(rewriter, lockOp->getLoc())({lockPtr});
      auto inflight = rewriter.createInFlight(mapping);
      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lockOp.getNested().front(), inflight.getRes(), [&](tuples::ReturnOpAdaptor adaptor) {
         if (!adaptor.getResults().empty()) {
            lockOp.getRes().replaceAllUsesWith(adaptor.getResults()[0]);
            rewriter.eraseOp(lockOp);
         } else {
            rewriter.eraseOp(lockOp);
         }
      });
      rt::EntryLock::unlock(rewriter, lockOp->getLoc())({lockPtr});
      return success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower