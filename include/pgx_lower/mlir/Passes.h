#ifndef MLIR_PASSES_H
#define MLIR_PASSES_H

#include "mlir/Pass/PassManager.h"
#include <memory>

namespace mlir {
namespace pgx_lower {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all pgx-lower conversion and transformation passes
void registerAllPgxLoweringPasses();

//===----------------------------------------------------------------------===//
// Pipeline Configuration  
//===----------------------------------------------------------------------===//

/// Create the complete RelAlg → DB → DSA lowering pipeline for Test 1
/// This pipeline handles the full transformation from RelAlg dialect to DSA dialect
/// 
/// Pipeline stages:
/// 1. RelAlg → DB lowering (Phase 3a)
/// 2. DB → DSA lowering (Phase 3b)
/// 
/// @param pm The pass manager to populate with passes
void createRelAlgToDBPipeline(mlir::PassManager& pm);


/// Create the complete lowering pipeline from RelAlg to DSA
/// This is the main pipeline for Test 1 execution
/// 
/// @param pm The pass manager to populate with passes
/// @param enableVerifier Whether to enable MLIR verification between passes
void createCompleteLoweringPipeline(mlir::PassManager& pm, bool enableVerifier = true);

//===----------------------------------------------------------------------===//
// Future Pipeline Extensions (Phase 4+)
//===----------------------------------------------------------------------===//

// TODO: Add DSA → LLVM pipeline configuration (Phase 4)
// void createDSAToLLVMPipeline(mlir::PassManager& pm);

// TODO: Add optimization pipeline configuration
// void createOptimizationPipeline(mlir::PassManager& pm);

//===----------------------------------------------------------------------===//
// Library Validation Functions
//===----------------------------------------------------------------------===//

/// Validate that all required MLIR libraries and passes are properly loaded
/// This helps detect issues with symbol resolution and dynamic library loading
/// 
/// @return true if all validations pass, false otherwise
bool validateLibraryLoading();

/// Validate that all required dialects are properly registered
/// Checks for dialect TypeID registration and operation availability
/// 
/// @return true if all dialects are available, false otherwise
bool validateDialectRegistration();

/// Validate that all conversion passes are properly registered
/// Ensures pass registration completed successfully
/// 
/// @return true if all passes are registered, false otherwise  
bool validatePassRegistration();

} // namespace pgx_lower
} // namespace mlir

#endif // MLIR_PASSES_H