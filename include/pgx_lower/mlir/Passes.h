#ifndef MLIR_PASSES_H
#define MLIR_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
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
void createRelAlgToDBPipeline(::mlir::PassManager& pm);


/// DEPRECATED: Single PassManager approach - use runCompleteLoweringPipeline instead
/// This function is maintained for backwards compatibility but is not recommended
/// 
/// @param pm The pass manager to populate with passes
/// @param enableVerifier Whether to enable MLIR verification between passes
void createCompleteLoweringPipeline(::mlir::PassManager& pm, bool enableVerifier = true);

/// LingoDB-Compliant Multi-PassManager Pipeline (RECOMMENDED)
/// Runs the complete lowering pipeline using 3 separate PassManager instances
/// following LingoDB's proven architecture pattern
/// 
/// Pipeline phases:
/// 1. RelAlg → Mixed DB+DSA+Util (Function-Scoped with Nested Pass)
/// 2. DB → Standard MLIR + PostgreSQL SPI (Module-Scoped)
/// 3. DSA+Util → Standard MLIR/LLVM (Module-Scoped)
/// 
/// @param module The MLIR module to transform
/// @param context The MLIR context
/// @param enableVerifier Whether to enable MLIR verification between passes
/// @return LogicalResult success() if all phases succeed, failure() otherwise
LogicalResult runCompleteLoweringPipeline(::mlir::ModuleOp module, ::mlir::MLIRContext* context, bool enableVerifier = true);

//===----------------------------------------------------------------------===//
// Future Pipeline Extensions (Phase 4+)
//===----------------------------------------------------------------------===//

// TODO: Add DSA → LLVM pipeline configuration (Phase 4)
// void createDSAToLLVMPipeline(::mlir::PassManager& pm);

// TODO: Add optimization pipeline configuration
// void createOptimizationPipeline(::mlir::PassManager& pm);

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