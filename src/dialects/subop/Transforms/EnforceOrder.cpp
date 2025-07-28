#include "dialects/subop/Transforms/Passes.h"
#include "dialects/subop/Transforms/SubOpDependencyAnalysis.h"

#include "mlir/IR/BuiltinOps.h"

namespace {
using namespace pgx_lower::compiler::dialect;

class EnforceOrderPass : public mlir::PassWrapper<EnforceOrderPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnforceOrderPass)
   virtual llvm::StringRef getArgument() const override { return "subop-enforce-order"; }

   void runOnOperation() override {
      auto subOpDependencyAnalysis = getAnalysis<subop::SubOpDependencyAnalysis>();
      std::vector<std::pair<mlir::Operation*, mlir::Operation*>> otherOrdering;
      for (auto& localOrder : subOpDependencyAnalysis.validOrder) {
         mlir::Operation* last = nullptr;
         for (auto* x : localOrder.second) {
            if (!last) {
               x->moveAfter(localOrder.first, localOrder.first->begin());
            } else {
               x->moveAfter(last);
            }
            last = x;
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> subop::createEnforceOrderPass() { return std::make_unique<EnforceOrderPass>(); }
