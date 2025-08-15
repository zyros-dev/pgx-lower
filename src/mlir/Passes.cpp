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
#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"

namespace mlir {
namespace pgx_lower {

void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification) {
    // Phase 4d Architecture: RelAlg → (DB + DSA + Util) → Standard MLIR → LLVM
    // Following LingoDB's unified PassManager pattern
    
    PGX_DEBUG("createCompleteLoweringPipeline: Building unified lowering pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Phase 1: RelAlg→DB+DSA+Util (generates mixed dialects)
    PGX_DEBUG("Adding RelAlg→DB pass");
    pm.addNestedPass<func::FuncOp>(relalg::createLowerToDBPass());
    
    // Phase 2: DB+DSA→Standard (parallel lowering of mixed dialects)
    PGX_DEBUG("Adding DB→Std pass");
    pm.addPass(db::createLowerToStdPass());
    
    PGX_DEBUG("Adding DSA→Std pass");
    pm.addPass(dsa::createLowerToStdPass());
    
    // Add canonicalizer to clean up
    pm.addPass(createCanonicalizerPass());
    
    // Phase 3: Standard→LLVM
    PGX_DEBUG("Adding Standard→LLVM conversion passes");
    pm.addPass(createConvertSCFToCFPass());
    // Util patterns are included in the unified Standard→LLVM pass
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    // Phase 4: Function-level optimizations
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    
    PGX_DEBUG("createCompleteLoweringPipeline: Unified pipeline configured");
}

// Phase 1: RelAlg→DB lowering pipeline
void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification) {
    PGX_DEBUG("createRelAlgToDBPipeline: Starting Phase 1 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add RelAlg to DB lowering pass
    PGX_DEBUG("createRelAlgToDBPipeline: Creating RelAlg to DB pass...");
    auto pass = relalg::createLowerToDBPass();
    if (!pass) {
        PGX_ERROR("createRelAlgToDBPipeline: Failed to create RelAlg to DB pass!");
        return;
    }
    PGX_DEBUG("createRelAlgToDBPipeline: Pass created successfully, adding to pipeline...");
    pm.addNestedPass<func::FuncOp>(std::move(pass));
    
    PGX_DEBUG("createRelAlgToDBPipeline: Phase 1 pipeline configured");
}

// Phase 2: DB+DSA→Standard lowering pipeline  
void createDBDSAToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("createDBDSAToStandardPipeline: ENTRY - Starting Phase 2 pipeline configuration");
    PGX_DEBUG("createDBDSAToStandardPipeline: Starting Phase 2 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Enhanced pass creation with crash isolation
    PGX_INFO("createDBDSAToStandardPipeline: BEFORE DB pass creation");
    auto dbPass = db::createLowerToStdPass();
    if (!dbPass) {
        PGX_ERROR("createDBDSAToStandardPipeline: DB pass creation returned null!");
        return;
    }
    PGX_INFO("createDBDSAToStandardPipeline: BEFORE adding DB pass to pipeline");
    pm.addPass(std::move(dbPass));
    PGX_INFO("createDBDSAToStandardPipeline: DB pass added to pipeline successfully");
    
    // Add canonicalizer between passes to clean up and validate state
    PGX_DEBUG("createDBDSAToStandardPipeline: Adding first canonicalizer pass");
    pm.addPass(createCanonicalizerPass());
    
    // Second: DSA→Std pass
    PGX_INFO("createDBDSAToStandardPipeline: BEFORE DSA pass creation");
    auto dsaPass = dsa::createLowerToStdPass();
    if (!dsaPass) {
        PGX_ERROR("createDBDSAToStandardPipeline: DSA pass creation returned null!");
        return;
    }
    PGX_INFO("createDBDSAToStandardPipeline: DSA pass created successfully");
    
    PGX_INFO("createDBDSAToStandardPipeline: BEFORE adding DSA pass to pipeline");
    pm.addPass(std::move(dsaPass));
    PGX_INFO("createDBDSAToStandardPipeline: DSA pass added to pipeline successfully");
    
    // Final canonicalizer
    PGX_DEBUG("createDBDSAToStandardPipeline: Adding final canonicalizer pass");
    pm.addPass(createCanonicalizerPass());
    
    PGX_DEBUG("createDBDSAToStandardPipeline: Phase 2 pipeline configured with sequential execution");
}

// Phase 3: Standard→LLVM lowering pipeline
void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    PGX_DEBUG("createStandardToLLVMPipeline: Starting Phase 3 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Convert SCF to ControlFlow first (like LingoDB)
    pm.addPass(createConvertSCFToCFPass());
    
    // Add all Standard→LLVM conversion passes
    // Util patterns are included in the unified Standard→LLVM pass
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    
    // Reconcile any unrealized casts
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    PGX_DEBUG("createStandardToLLVMPipeline: Phase 3 pipeline configured");
}

// Helper to run function-level optimizations (like LingoDB's pmFunc)
void createFunctionOptimizationPipeline(PassManager& pm, bool enableVerification) {
    PGX_DEBUG("createFunctionOptimizationPipeline: Starting function-level optimizations");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add function-level optimization passes
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createCSEPass());
    
    PGX_DEBUG("createFunctionOptimizationPipeline: Function optimizations configured");
}

} // namespace pgx_lower
} // namespace mlir