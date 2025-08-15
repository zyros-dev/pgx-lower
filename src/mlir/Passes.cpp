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
    PGX_INFO("createRelAlgToDBPipeline: Starting Phase 1 pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add RelAlg to DB lowering pass
    PGX_INFO("createRelAlgToDBPipeline: Creating RelAlg to DB pass...");
    auto pass = relalg::createLowerToDBPass();
    if (!pass) {
        PGX_ERROR("createRelAlgToDBPipeline: Failed to create RelAlg to DB pass!");
        return;
    }
    PGX_INFO("createRelAlgToDBPipeline: Pass created successfully, adding to pipeline...");
    pm.addNestedPass<func::FuncOp>(std::move(pass));
    
    PGX_INFO("createRelAlgToDBPipeline: Phase 1 pipeline configured");
}

// Phase 2: DB+DSA→Standard lowering pipeline  
void createDBDSAToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("createDBDSAToStandardPipeline: ENTRY - Starting Phase 2 pipeline configuration");
    PGX_INFO("createDBDSAToStandardPipeline: Starting Phase 2 pipeline");
    
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
    PGX_INFO("createDBDSAToStandardPipeline: Adding first canonicalizer pass");
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
    PGX_INFO("createDBDSAToStandardPipeline: Adding final canonicalizer pass");
    pm.addPass(createCanonicalizerPass());
    
    PGX_INFO("createDBDSAToStandardPipeline: Phase 2 pipeline configured with sequential execution");
}

void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Use the working StandardToLLVMPass from StandardToLLVM.cpp
    PGX_INFO("createStandardToLLVMPipeline: Using StandardToLLVMPass");

    pm.addPass(std::move(createStandardToLLVMPass()));

    PGX_INFO("createStandardToLLVMPipeline: StandardToLLVMPass added successfully");
}

} // namespace pgx_lower
} // namespace mlir