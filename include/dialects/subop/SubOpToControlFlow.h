#ifndef PGX_LOWER_COMPILER_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
#define PGX_LOWER_COMPILER_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

#include "Transforms/Passes.h"

namespace pgx_lower::compiler::dialect {
namespace subop {
void setCompressionEnabled(bool compressionEnabled);
std::unique_ptr<mlir::Pass> createLowerSubOpPass();
std::unique_ptr<mlir::Pass> createLowerSubOpToControlFlowPass(); // New stub implementation
std::unique_ptr<mlir::Pass> createSimpleSubOpToControlFlowPass(); // Simplified implementation
std::unique_ptr<mlir::Pass> createMinimalSubOpToControlFlowPass(); // Minimal working implementation
void registerSubOpToControlFlowConversionPasses();
void createLowerSubOpPipeline(mlir::OpPassManager& pm);
} // end namespace subop
} // end namespace pgx_lower::compiler::dialect
#endif //PGX_LOWER_COMPILER_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
