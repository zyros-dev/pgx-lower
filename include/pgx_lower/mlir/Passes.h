#ifndef PGX_LOWER_MLIR_PASSES_H
#define PGX_LOWER_MLIR_PASSES_H

namespace mlir {
class PassManager;

namespace pgx_lower {

// DEPRECATED: Use sequential PassManagers instead
// This function exists for backward compatibility but should not be used
void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification = false);

// Sequential PassManager approach following LingoDB patterns (runner.cpp:413-447)
// Phase 1: RelAlg→DB lowering pipeline
void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification = false);

// Phase 2: DB+DSA→Standard lowering pipeline
void createDBDSAToStandardPipeline(PassManager& pm, bool enableVerification = false);

// Phase 3: Standard→LLVM lowering pipeline
void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification = false);

// Helper: Function-level optimization pipeline
void createFunctionOptimizationPipeline(PassManager& pm, bool enableVerification = false);

} // namespace pgx_lower
} // namespace mlir

#endif // PGX_LOWER_MLIR_PASSES_H