#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "execution/logging.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

namespace mlir {
namespace pgx_lower {

void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification) {
    // Phase 4d Architecture: RelAlg → (DB + DSA + Util) → Standard MLIR → LLVM
    
    PGX_DEBUG("createCompleteLoweringPipeline called with enableVerification=" + std::to_string(enableVerification));
    
    if (enableVerification) {
        PGX_DEBUG("Enabling MLIR verifier");
        pm.enableVerifier(true);
    }
    
    // Phase 1: RelAlg to mixed DB+DSA+Util operations
    PGX_DEBUG("Phase 1: Adding RelAlg to DB lowering pass");
    try {
        pm.addNestedPass<func::FuncOp>(relalg::createLowerToDBPass());
        PGX_DEBUG("Successfully added RelAlg to DB pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add RelAlg to DB pass: " + std::string(e.what()));
        throw;
    }
    
    // Phase 2: Parallel lowering to Standard MLIR
    PGX_DEBUG("Phase 2: Adding parallel lowering passes (DB→Std, DSA→Std)");
    
    // Run each pass in its own PassManager to isolate the crash
    PGX_DEBUG("Creating DB→Std pass");
    try {
        pm.addPass(db::createLowerToStdPass());
        PGX_DEBUG("Successfully added DB→Std pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add DB→Std pass: " + std::string(e.what()));
        throw;
    }
    
    PGX_DEBUG("Creating DSA→Std pass");
    try {
        pm.addPass(dsa::createLowerToStdPass());
        PGX_DEBUG("Successfully added DSA→Std pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add DSA→Std pass: " + std::string(e.what()));
        throw;
    }
    // Note: UtilToLLVM is added in Phase 3 as it goes directly to LLVM
    
    // Phase 3: Standard MLIR to LLVM conversion
    PGX_DEBUG("Phase 3: Adding Standard→LLVM conversion passes");
    
    // Note: UtilToLLVM patterns are included in the FuncToLLVM pass
    // No separate pass needed for Util dialect operations
    
    // Standard to LLVM conversion passes
    PGX_DEBUG("Adding SCF to ControlFlow pass");
    try {
        pm.addPass(createConvertSCFToCFPass());
        PGX_DEBUG("Successfully added SCF to ControlFlow pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add SCF to ControlFlow pass: " + std::string(e.what()));
        throw;
    }
    
    PGX_DEBUG("Adding Func to LLVM pass");
    try {
        pm.addPass(createConvertFuncToLLVMPass());
        PGX_DEBUG("Successfully added Func to LLVM pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add Func to LLVM pass: " + std::string(e.what()));
        throw;
    }
    
    PGX_DEBUG("Adding Arith to LLVM pass");
    try {
        pm.addPass(createArithToLLVMConversionPass());
        PGX_DEBUG("Successfully added Arith to LLVM pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add Arith to LLVM pass: " + std::string(e.what()));
        throw;
    }
    
    PGX_DEBUG("Adding ControlFlow to LLVM pass");
    try {
        pm.addPass(createConvertControlFlowToLLVMPass());
        PGX_DEBUG("Successfully added ControlFlow to LLVM pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add ControlFlow to LLVM pass: " + std::string(e.what()));
        throw;
    }
    
    PGX_DEBUG("Adding reconcile unrealized casts pass");
    try {
        pm.addPass(createReconcileUnrealizedCastsPass());
        PGX_DEBUG("Successfully added reconcile unrealized casts pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add reconcile unrealized casts pass: " + std::string(e.what()));
        throw;
    }
    
    // Phase 4: Final optimizations
    PGX_DEBUG("Phase 4: Adding final optimization passes");
    try {
        pm.addPass(createCanonicalizerPass());
        PGX_DEBUG("Successfully added canonicalizer pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add canonicalizer pass: " + std::string(e.what()));
        throw;
    }
    
    try {
        pm.addPass(createCSEPass());
        PGX_DEBUG("Successfully added CSE pass");
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to add CSE pass: " + std::string(e.what()));
        throw;
    }
    
    PGX_DEBUG("createCompleteLoweringPipeline completed successfully - all passes added");
}

} // namespace pgx_lower
} // namespace mlir