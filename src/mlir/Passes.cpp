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
    // Following LingoDB's proven sequential PassManager pattern (runner.cpp:413-447)
    
    PGX_ERROR("createCompleteLoweringPipeline: DEPRECATED - Use sequential PassManagers");
    
    // This function is deprecated in favor of sequential PassManager approach
    // See mlir_runner.cpp for the correct implementation pattern
    assert(false && "Use sequential PassManagers instead of unified pipeline");
}

// Phase 1: RelAlg→DB lowering pipeline
void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification) {
    PGX_DEBUG("createRelAlgToDBPipeline: Starting Phase 1 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add RelAlg to DB lowering pass
    pm.addNestedPass<func::FuncOp>(relalg::createLowerToDBPass());
    
    PGX_DEBUG("createRelAlgToDBPipeline: Phase 1 pipeline configured");
}

// Phase 2: DB+DSA→Standard lowering pipeline  
void createDBDSAToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_DEBUG("createDBDSAToStandardPipeline: Starting Phase 2 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add DB→Std and DSA→Std passes (module-level)
    pm.addPass(db::createLowerToStdPass());
    pm.addPass(dsa::createLowerToStdPass());
    
    // Add canonicalizer like LingoDB does
    pm.addPass(createCanonicalizerPass());
    
    PGX_DEBUG("createDBDSAToStandardPipeline: Phase 2 pipeline configured");
}

// Phase 3: Standard→LLVM lowering pipeline
void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    PGX_DEBUG("createStandardToLLVMPipeline: Starting Phase 3 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Convert SCF to ControlFlow first (like LingoDB)
    pm.addPass(createConvertSCFToCFPass());
    
    // Use our unified Standard→LLVM pass that includes all patterns
    pm.addPass(createStandardToLLVMPass());
    
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