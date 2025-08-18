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

// Additional includes for unified conversion pass
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pgx_lower {

// Forward declarations
std::unique_ptr<Pass> createConvertToLLVMPass();
std::unique_ptr<Pass> createStandardToLLVMPass();


// Phase 1: RelAlg→DB lowering pipeline
void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("createRelAlgToDBPipeline: Adding relalg to db pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    auto pass = relalg::createLowerToDBPass();
    if (!pass) {
        PGX_ERROR("createRelAlgToDBPipeline: Failed to create RelAlg to DB pass!");
        return;
    }
    pm.addNestedPass<func::FuncOp>(std::move(pass));
    pm.addPass(createCanonicalizerPass());
}

// Phase 2a: DB→Standard lowering pipeline (following LingoDB sequential pattern)
void createDBToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("[createDBToStandardPipeline]: Adding DB to Standard pipeline (Phase 2a)");
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    auto dbPass = db::createLowerToStdPass();
    if (!dbPass) {
        PGX_ERROR("createDBToStandardPipeline: DB pass creation returned null!");
        return;
    }
    pm.addPass(std::move(dbPass));
    pm.addPass(createCanonicalizerPass());
}

// Phase 2b: DSA→Standard lowering pipeline (following LingoDB sequential pattern)
void createDSAToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("[createDSAToStandardPipeline]: Adding DSA to Standard pipeline (Phase 2b)");
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    auto dsaPass = dsa::createLowerToStdPass();
    if (!dsaPass) {
        PGX_ERROR("createDSAToStandardPipeline: DSA pass creation returned null!");
        return;
    }
    pm.addPass(std::move(dsaPass));
    pm.addPass(createCanonicalizerPass());
}

void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("[createStandardToLLVMPipeline]: Adding DB DSA to Standard pipeline");
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    pm.addPass(std::move(createStandardToLLVMPass()));
    pm.addPass(createCanonicalizerPass());
}

} // namespace pgx_lower
} // namespace mlir