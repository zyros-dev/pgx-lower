#ifndef PGX_LOWER_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
#define PGX_LOWER_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace pgx_lower::compiler::dialect {
namespace subop {
std::unique_ptr<mlir::Pass> createFoldColumnsPass();
std::unique_ptr<mlir::Pass> createEnforceOrderPass();
std::unique_ptr<mlir::Pass> createNormalizeSubOpPass();
std::unique_ptr<mlir::Pass> createSpecializeSubOpPass(bool withOptimizations);
std::unique_ptr<mlir::Pass> createPullGatherUpPass();
std::unique_ptr<mlir::Pass> createReuseLocalPass();
std::unique_ptr<mlir::Pass> createGlobalOptPass();
std::unique_ptr<mlir::Pass> createParallelizePass();
std::unique_ptr<mlir::Pass> createSpecializeParallelPass();
std::unique_ptr<mlir::Pass> createSplitIntoExecutionStepsPass();
std::unique_ptr<mlir::Pass> createInlineNestedMapPass();
std::unique_ptr<mlir::Pass> createFinalizePass();
std::unique_ptr<mlir::Pass> createPrepareLoweringPass();
void registerSubOpTransformations();
} // end namespace subop
} // end namespace pgx_lower::compiler::dialect

#endif //PGX_LOWER_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
