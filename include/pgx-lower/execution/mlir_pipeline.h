#ifndef MLIR_PIPELINE_H
#define MLIR_PIPELINE_H

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace mlir_runner {

bool setupMLIRContextForJIT(::mlir::MLIRContext& context);
bool runPhase3a(::mlir::ModuleOp module);
bool runPhase3b(::mlir::ModuleOp module);
bool runPhase3c(::mlir::ModuleOp module);
bool runCompleteLoweringPipeline(::mlir::ModuleOp module);
void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase);
bool validateModuleState(::mlir::ModuleOp module, const std::string& phase);

} // namespace mlir_runner

#endif // MLIR_PIPELINE_H