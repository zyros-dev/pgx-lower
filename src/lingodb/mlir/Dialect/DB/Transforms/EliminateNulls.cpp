#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <iostream>

#include "lingodb/mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
#include "EliminateNulls.inc"

class WrapWithNullCheck : public mlir::RewritePattern {
   public:
   WrapWithNullCheck(::mlir::MLIRContext* context) : RewritePattern(MatchAnyOpTypeTag(), mlir::PatternBenefit(1), context) {}
   ::mlir::LogicalResult match(::mlir::Operation* op) const override {
      if (op->getNumResults() > 1) return mlir::failure();
      if (op->getNumResults() == 1 && !op->getResultTypes()[0].isa<mlir::db::NullableType>()) return mlir::failure();
      auto needsWrapInterface = mlir::dyn_cast_or_null<mlir::db::NeedsNullWrap>(op);
      if (!needsWrapInterface) return mlir::failure();
      if (!needsWrapInterface.needsNullWrap()) return mlir::failure();
      if (llvm::any_of(op->getOperands(), [](::mlir::Value v) { return v.getType().isa<mlir::db::NullableType>(); })) {
         return mlir::success();
      }
      return mlir::failure();
   }

   void rewrite(::mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      rewriter.setInsertionPoint(op);
      ::mlir::Value isAnyNull;
      for (auto operand : op->getOperands()) {
         if (operand.getType().isa<mlir::db::NullableType>()) {
            auto isCurrNull = rewriter.create<mlir::db::IsNullOp>(op->getLoc(), operand);
            if (isAnyNull) {
               isAnyNull = rewriter.create<mlir::arith::OrIOp>(op->getLoc(), isAnyNull, isCurrNull);
            } else {
               isAnyNull = isCurrNull;
            }
         }
      }

      auto supInvVal = mlir::dyn_cast_or_null<mlir::db::SupportsInvalidValues>(op);
      if (supInvVal && supInvVal.supportsInvalidValues()) {
         ::mlir::IRMapping mapping;
         for (auto operand : op->getOperands()) {
            if (operand.getType().isa<mlir::db::NullableType>()) {
               mapping.map(operand, rewriter.create<mlir::db::NullableGetVal>(op->getLoc(), operand));
            }
         }
         auto* cloned = rewriter.clone(*op, mapping);
         if (op->getNumResults() == 1) {
            cloned->getResult(0).setType(getBaseType(cloned->getResult(0).getType()));
            rewriter.replaceOpWithNewOp<mlir::db::AsNullableOp>(op, op->getResultTypes()[0], cloned->getResult(0), isAnyNull);
         } else {
            rewriter.eraseOp(op);
         }
         return;
      } else {
         auto ifOp = rewriter.create<mlir::scf::IfOp>(op->getLoc(), op->getResultTypes(), isAnyNull, true);
         {
            // Then branch - handle null case
            auto& thenRegion = ifOp.getThenRegion();
            thenRegion.push_back(new mlir::Block());
            rewriter.setInsertionPointToStart(&thenRegion.front());
            if(op->getNumResults()==1){
               ::mlir::Value nullResult = rewriter.create<mlir::db::NullOp>(op->getLoc(), op->getResultTypes()[0]);
               rewriter.create<mlir::scf::YieldOp>(op->getLoc(), nullResult);
            }else{
               rewriter.create<mlir::scf::YieldOp>(op->getLoc());
            }
         }
         {
            // Else branch - normal processing
            auto& elseRegion = ifOp.getElseRegion();
            elseRegion.push_back(new mlir::Block());
            rewriter.setInsertionPointToStart(&elseRegion.front());
            ::mlir::IRMapping mapping;
            for (auto operand : op->getOperands()) {
               if (operand.getType().isa<mlir::db::NullableType>()) {
                  mapping.map(operand, rewriter.create<mlir::db::NullableGetVal>(op->getLoc(), operand));
               }
            }
            auto *cloned = rewriter.clone(*op, mapping);
            if(op->getNumResults()==1){
               cloned->getResult(0).setType(getBaseType(cloned->getResult(0).getType()));
               ::mlir::Value nullResult = rewriter.create<mlir::db::AsNullableOp>(op->getLoc(), op->getResultTypes()[0], cloned->getResult(0));
               rewriter.create<mlir::scf::YieldOp>(op->getLoc(), nullResult);
            }else{
               rewriter.create<mlir::scf::YieldOp>(op->getLoc());
            }
         }
         rewriter.replaceOp(op, ifOp);
      }
   }
};

//Pattern that optimizes the join order
class EliminateNulls : public ::mlir::OperationPass<::mlir::ModuleOp> {
   virtual llvm::StringRef getArgument() const override { return "eliminate-nulls"; }
   virtual llvm::StringRef getName() const override { return getArgument(); }
   std::unique_ptr<Pass> clonePass() const override { return std::make_unique<EliminateNulls>(*this); }
public:
   EliminateNulls() : ::mlir::OperationPass<::mlir::ModuleOp>(::mlir::TypeID::get<EliminateNulls>()) {}
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
   }

   public:
   void runOnOperation() override {
      {
         mlir::RewritePatternSet patterns(&getContext());
         //patterns.insert<EliminateNullCmp>(&getContext());
         patterns.insert<EliminateDeriveTruthNonNullable>(&getContext());
         patterns.insert<EliminateDeriveTruthNullable>(&getContext());
         patterns.insert<WrapWithNullCheck>(&getContext());
         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir::db {

std::unique_ptr<Pass> createEliminateNullsPass() { return std::make_unique<EliminateNulls>(); }

} // end namespace mlir::db
