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
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Phase 1: RelAlg to mixed DB+DSA+Util operations
    PGX_DEBUG("Adding RelAlg to DB lowering pass");
    pm.addNestedPass<func::FuncOp>(relalg::createLowerToDBPass());
    
    // Phase 2: Parallel lowering to Standard MLIR
    PGX_DEBUG("Adding parallel lowering passes (DB→Std, DSA→Std)");
    pm.addPass(db::createLowerToStdPass());
    pm.addPass(dsa::createLowerToStdPass());
    
    // Phase 3: Standard MLIR to LLVM conversion
    PGX_DEBUG("Adding Standard→LLVM conversion passes");
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    // Phase 4: Final optimizations
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
}

} // namespace pgx_lower
} // namespace mlir