#ifndef MLIR_PIPELINE_H
#define MLIR_PIPELINE_H

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace mlir_runner {

// MLIR Context Management - PostgreSQL-free
bool setupMLIRContextForJIT(::mlir::MLIRContext& context);

// Phase 3a: RelAlg → DB+DSA+Util lowering
bool runPhase3a(::mlir::ModuleOp module);

// Phase 3b: DB+DSA+Util → Standard lowering  
bool runPhase3b(::mlir::ModuleOp module);

// Phase 3c: Standard → LLVM lowering
bool runPhase3c(::mlir::ModuleOp module);

// Complete pipeline: RelAlg → DB+DSA+Util → Standard → LLVM
bool runCompleteLoweringPipeline(::mlir::ModuleOp module);

// Utility functions for debugging and analysis
void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title);
bool validateModuleState(::mlir::ModuleOp module, const std::string& phase);

} // namespace mlir_runner

#endif // MLIR_PIPELINE_H