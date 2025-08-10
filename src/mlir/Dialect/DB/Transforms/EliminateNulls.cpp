#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include <iostream>

// #include "mlir/Dialect/RelAlg/Passes.h"  // Not used
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
#include "EliminateNulls.inc"

class WrapWithNullCheck : public mlir::RewritePattern {
   public:
   WrapWithNullCheck(mlir::MLIRContext* context) : RewritePattern(MatchAnyOpTypeTag(), mlir::PatternBenefit(1), context) {}
   mlir::LogicalResult match(mlir::Operation* op) const override {
      if (op->getNumResults() > 1) return mlir::failure();
      if (op->getNumResults() == 1 && !op->getResultTypes()[0].isa<pgx::mlir::db::NullableType>()) return mlir::failure();
      auto needsWrapInterface = mlir::dyn_cast_or_null<pgx::mlir::db::NeedsNullWrap>(op);
      if (!needsWrapInterface) return mlir::failure();
      if (!needsWrapInterface.needsNullWrap()) return mlir::failure();
      if (llvm::any_of(op->getOperands(), [](mlir::Value v) { return v.getType().isa<pgx::mlir::db::NullableType>(); })) {
         return mlir::success();
      }
      return mlir::failure();
   }

   void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      rewriter.setInsertionPoint(op);
      mlir::Value isAnyNull;
      for (auto operand : op->getOperands()) {
         if (operand.getType().isa<pgx::mlir::db::NullableType>()) {
            auto isCurrNull = rewriter.create<pgx::mlir::db::IsNullOp>(op->getLoc(), operand);
            if (isAnyNull) {
               isAnyNull = rewriter.create<mlir::arith::OrIOp>(op->getLoc(), isAnyNull, isCurrNull);
            } else {
               isAnyNull = isCurrNull;
            }
         }
      }

      auto supInvVal = mlir::dyn_cast_or_null<pgx::mlir::db::SupportsInvalidValues>(op);
      if (supInvVal && supInvVal.supportsInvalidValues()) {
         mlir::IRMapping mapping;
         for (auto operand : op->getOperands()) {
            if (operand.getType().isa<pgx::mlir::db::NullableType>()) {
               mapping.map(operand, rewriter.create<pgx::mlir::db::NullableGetVal>(op->getLoc(), operand));
            }
         }
         auto* cloned = rewriter.clone(*op, mapping);
         if (op->getNumResults() == 1) {
            cloned->getResult(0).setType(getBaseType(cloned->getResult(0).getType()));
            rewriter.replaceOpWithNewOp<pgx::mlir::db::AsNullableOp>(op, op->getResultTypes()[0], cloned->getResult(0), isAnyNull);
         } else {
            rewriter.eraseOp(op);
         }
         return;
      } else {
         // LLVM 20 API: Create IfOp with builders using the new API
         auto thenBuilder = [&](mlir::OpBuilder& b, mlir::Location loc) {
            if(op->getNumResults()==1){
               mlir::Value nullResult=b.create<pgx::mlir::db::NullOp>(op->getLoc(),op->getResultTypes()[0]);
               b.create<mlir::scf::YieldOp>(loc,nullResult);
            }else{
               b.create<mlir::scf::YieldOp>(loc);
            }
         };
         auto elseBuilder = [&](mlir::OpBuilder& b, mlir::Location loc) {
            mlir::IRMapping mapping;
            for (auto operand : op->getOperands()) {
               if (operand.getType().isa<pgx::mlir::db::NullableType>()) {
                  mapping.map(operand,b.create<pgx::mlir::db::NullableGetVal>(op->getLoc(),operand));
               }
            }
            auto *cloned=b.clone(*op,mapping);
            if(op->getNumResults()==1){
               cloned->getResult(0).setType(getBaseType(cloned->getResult(0).getType()));
               mlir::Value nullResult=b.create<pgx::mlir::db::AsNullableOp>(op->getLoc(),op->getResultTypes()[0],cloned->getResult(0));
               b.create<mlir::scf::YieldOp>(loc,nullResult);
            }else{
               b.create<mlir::scf::YieldOp>(loc);
            }
         };
         
         // LLVM 20: Use the correct API to create IfOp
         auto ifOp = rewriter.create<mlir::scf::IfOp>(op->getLoc(), isAnyNull, thenBuilder, elseBuilder);
         rewriter.replaceOp(op, ifOp.getResults());
      }
   }
};

//Pattern that optimizes the join order
class EliminateNulls : public mlir::PassWrapper<EliminateNulls, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "eliminate-nulls"; }
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
   }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         //patterns.insert<EliminateNullCmp>(&getContext());
         patterns.insert<EliminateDeriveTruthNonNullable>(&getContext());
         patterns.insert<EliminateDeriveTruthNullable>(&getContext());
         //patterns.insert<SimplifyNullableCondSkip>(&getContext());
         patterns.insert<WrapWithNullCheck>(&getContext());
         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace pgx::mlir::db {

std::unique_ptr<::mlir::Pass> createEliminateNullsPass() { return std::make_unique<EliminateNulls>(); }

} // end namespace pgx::mlir::db
