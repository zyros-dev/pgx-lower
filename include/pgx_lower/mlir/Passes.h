#ifndef PGX_LOWER_MLIR_PASSES_H
#define PGX_LOWER_MLIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class PassManager;

namespace pgx_lower {

// DEPRECATED: Use sequential PassManagers instead
// This function exists for backward compatibility but should not be used
void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification = false);

// Sequential PassManager approach following LingoDB patterns (runner.cpp:413-447)
// Phase 1: RelAlg→DB lowering pipeline
void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification = false);

// Phase 2a: DB→Standard lowering pipeline (sequential approach)
void createDBToStandardPipeline(PassManager& pm, bool enableVerification = false);

// Phase 2b: DSA→Standard lowering pipeline (sequential approach)  
void createDSAToStandardPipeline(PassManager& pm, bool enableVerification = false);

// Phase 3: Standard→LLVM lowering pipeline
void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification = false);

// Unified Standard→LLVM conversion pass (LingoDB approach)
std::unique_ptr<Pass> createConvertToLLVMPass();

// Helper: Function-level optimization pipeline
void createFunctionOptimizationPipeline(PassManager& pm, bool enableVerification = false);

} // namespace pgx_lower
} // namespace mlir

#endif // PGX_LOWER_MLIR_PASSES_H