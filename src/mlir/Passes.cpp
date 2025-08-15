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

// Forward declaration for unified conversion pass
std::unique_ptr<Pass> createConvertToLLVMPass();


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

// Phase 3: Standard→LLVM lowering pipeline (LingoDB unified approach)
void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("createStandardToLLVMPipeline: ENTRY - Starting unified LingoDB-style conversion");
    
    if (enableVerification) {
        PGX_INFO("createStandardToLLVMPipeline: Enabling verifier");
        pm.enableVerifier(true);
        PGX_INFO("createStandardToLLVMPipeline: Verifier enabled successfully");
    }
    
    // Add unified conversion pass that handles all Standard→LLVM lowering
    // This matches LingoDB's approach exactly and avoids type converter conflicts
    PGX_INFO("createStandardToLLVMPipeline: BEFORE createConvertToLLVMPass()");
    auto pass = createConvertToLLVMPass();
    PGX_INFO("createStandardToLLVMPipeline: Pass created successfully");
    
    PGX_INFO("createStandardToLLVMPipeline: BEFORE pm.addPass()");
    pm.addPass(std::move(pass));
    PGX_INFO("createStandardToLLVMPipeline: Pass added to pipeline successfully");
    
    PGX_INFO("createStandardToLLVMPipeline: EXIT - Unified conversion pass configured successfully");
}

// Unified Standard→LLVM conversion pass implementation (based on LingoDB runner.cpp)
namespace {
struct ConvertToLLVMPass : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override {
        PGX_INFO("ConvertToLLVMPass: get depdendent dialects");
        registry.insert<LLVM::LLVMDialect>();
    }
    
    void runOnOperation() override {
        PGX_INFO("ConvertToLLVMPass: ENTRY - Starting unified conversion");
        auto module = getOperation();
        PGX_INFO("ConvertToLLVMPass: Got module operation successfully");
        auto* context = &getContext();
        PGX_INFO("ConvertToLLVMPass: Got context successfully");
        
        // Create DataLayoutAnalysis and LLVM type converter (like LingoDB)
        PGX_INFO("ConvertToLLVMPass: BEFORE DataLayoutAnalysis creation");
        DataLayoutAnalysis dataLayoutAnalysis(module);
        PGX_INFO("ConvertToLLVMPass: DataLayoutAnalysis created successfully");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE getAtOrAbove call");
        auto dataLayout = dataLayoutAnalysis.getAtOrAbove(module);
        PGX_INFO("ConvertToLLVMPass: getAtOrAbove completed successfully");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE LowerToLLVMOptions creation");
        LowerToLLVMOptions options(context, dataLayout);
        PGX_INFO("ConvertToLLVMPass: LowerToLLVMOptions created successfully");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE LLVMTypeConverter creation");
        LLVMTypeConverter typeConverter(context, options, &dataLayoutAnalysis);
        PGX_INFO("ConvertToLLVMPass: LLVMTypeConverter created successfully");
        
        // Add source materialization (like LingoDB)
        PGX_INFO("ConvertToLLVMPass: BEFORE source materialization setup");
        typeConverter.addSourceMaterialization([&](OpBuilder&, FunctionType type, ValueRange valueRange, Location loc) {
            PGX_INFO("ConvertToLLVMPass: Source materialization callback invoked");
            return valueRange.front();
        });
        PGX_INFO("ConvertToLLVMPass: Source materialization configured successfully");
        
        // Create unified pattern set
        PGX_INFO("ConvertToLLVMPass: BEFORE RewritePatternSet creation");
        RewritePatternSet patterns(context);
        PGX_INFO("ConvertToLLVMPass: RewritePatternSet created successfully");
        
        // Populate all conversion patterns in one unified set (LingoDB approach)  
        PGX_INFO("ConvertToLLVMPass: BEFORE populateAffineToStdConversionPatterns");
        populateAffineToStdConversionPatterns(patterns);
        PGX_INFO("ConvertToLLVMPass: populateAffineToStdConversionPatterns completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE populateSCFToControlFlowConversionPatterns");
        populateSCFToControlFlowConversionPatterns(patterns);
        PGX_INFO("ConvertToLLVMPass: populateSCFToControlFlowConversionPatterns completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE populateUtilToLLVMConversionPatterns");
        mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);  // Key addition!
        PGX_INFO("ConvertToLLVMPass: populateUtilToLLVMConversionPatterns completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE populateFuncToLLVMConversionPatterns");
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        PGX_INFO("ConvertToLLVMPass: populateFuncToLLVMConversionPatterns completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE populateControlFlowToLLVMConversionPatterns");
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        PGX_INFO("ConvertToLLVMPass: populateControlFlowToLLVMConversionPatterns completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE populateFinalizeMemRefToLLVMConversionPatterns");
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        PGX_INFO("ConvertToLLVMPass: populateFinalizeMemRefToLLVMConversionPatterns completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE populateArithToLLVMConversionPatterns");
        arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        PGX_INFO("ConvertToLLVMPass: populateArithToLLVMConversionPatterns completed");
        
        // Configure conversion target
        PGX_INFO("ConvertToLLVMPass: BEFORE ConversionTarget creation");
        ConversionTarget target(*context);
        PGX_INFO("ConvertToLLVMPass: ConversionTarget created successfully");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE target.addLegalDialect<LLVM::LLVMDialect>");
        target.addLegalDialect<LLVM::LLVMDialect>();
        PGX_INFO("ConvertToLLVMPass: target.addLegalDialect completed");
        
        PGX_INFO("ConvertToLLVMPass: BEFORE target.addLegalOp<ModuleOp>");
        target.addLegalOp<ModuleOp>();
        PGX_INFO("ConvertToLLVMPass: target.addLegalOp completed");
        
        // Apply unified conversion (like LingoDB)
        PGX_INFO("ConvertToLLVMPass: BEFORE applyFullConversion - THIS IS THE CRITICAL CALL");
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            PGX_ERROR("ConvertToLLVMPass: Unified conversion failed");
            signalPassFailure();
        } else {
            PGX_INFO("ConvertToLLVMPass: Unified conversion completed successfully");
        }
        PGX_INFO("ConvertToLLVMPass: EXIT - Conversion pass completed");
    }
};
} // end anonymous namespace

std::unique_ptr<Pass> createConvertToLLVMPass() {
    return std::make_unique<ConvertToLLVMPass>();
}


} // namespace pgx_lower
} // namespace mlir