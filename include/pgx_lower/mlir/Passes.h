#ifndef PGX_LOWER_MLIR_PASSES_H
#define PGX_LOWER_MLIR_PASSES_H

namespace mlir {
class PassManager;

namespace pgx_lower {

// Create the complete lowering pipeline for pgx-lower
// This implements the Phase 4d architecture:
// RelAlg → (DB + DSA + Util) → Standard MLIR → LLVM
void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification = false);

} // namespace pgx_lower
} // namespace mlir

#endif // PGX_LOWER_MLIR_PASSES_H