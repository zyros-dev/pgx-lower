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

void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification) {
    // Phase 4d Architecture: RelAlg → (DB + DSA + Util) → Standard MLIR → LLVM
    // Following LingoDB's unified PassManager pattern
    
    PGX_INFO("createCompleteLoweringPipeline: Building unified lowering pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Phase 1: RelAlg→DB+DSA+Util (generates mixed dialects)
    PGX_INFO("Adding RelAlg→DB pass");
    pm.addNestedPass<func::FuncOp>(relalg::createLowerToDBPass());
    
    // Phase 2: DB+DSA→Standard (parallel lowering of mixed dialects)
    PGX_INFO("Adding DB→Std pass");
    pm.addPass(db::createLowerToStdPass());
    
    PGX_INFO("Adding DSA→Std pass");
    pm.addPass(dsa::createLowerToStdPass());
    
    // Add canonicalizer to clean up
    pm.addPass(createCanonicalizerPass());
    
    // Phase 3: Standard→LLVM
    PGX_INFO("Adding Standard→LLVM conversion passes");
    pm.addPass(createConvertSCFToCFPass());
    // Util patterns are included in the unified Standard→LLVM pass
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    // Phase 4: Function-level optimizations
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    
    PGX_INFO("createCompleteLoweringPipeline: Unified pipeline configured");
}

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
    PGX_INFO("createStandardToLLVMPipeline: Starting unified LingoDB-style conversion");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add unified conversion pass that handles all Standard→LLVM lowering
    // This matches LingoDB's approach exactly and avoids type converter conflicts
    pm.addPass(createConvertToLLVMPass());
    
    PGX_INFO("createStandardToLLVMPipeline: Unified conversion pass added successfully");
}

// Unified Standard→LLVM conversion pass implementation (based on LingoDB runner.cpp)
namespace {
struct ConvertToLLVMPass : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override {
        PGX_INFO("ConvertToLLVMPass: get depdendent dialects");
        registry.insert<LLVM::LLVMDialect>();
    }
    
    void runOnOperation() override {
        PGX_INFO("ConvertToLLVMPass: Starting unified conversion");
        auto module = getOperation();
        auto* context = &getContext();
        
        // Create DataLayoutAnalysis and LLVM type converter (like LingoDB)
        DataLayoutAnalysis dataLayoutAnalysis(module);
        LowerToLLVMOptions options(context, dataLayoutAnalysis.getAtOrAbove(module));
        LLVMTypeConverter typeConverter(context, options, &dataLayoutAnalysis);
        
        // Add source materialization (like LingoDB)
        typeConverter.addSourceMaterialization([&](OpBuilder&, FunctionType type, ValueRange valueRange, Location loc) {
            return valueRange.front();
        });
        
        // Create unified pattern set
        RewritePatternSet patterns(context);
        
        // Populate all conversion patterns in one unified set (LingoDB approach)  
        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);  // Key addition!
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        
        // Configure conversion target
        ConversionTarget target(*context);
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalOp<ModuleOp>();
        
        // Apply unified conversion (like LingoDB)
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            PGX_ERROR("ConvertToLLVMPass: Unified conversion failed");
            signalPassFailure();
        } else {
            PGX_INFO("ConvertToLLVMPass: Unified conversion completed successfully");
        }
    }
};
} // end anonymous namespace

std::unique_ptr<Pass> createConvertToLLVMPass() {
    return std::make_unique<ConvertToLLVMPass>();
}

// Helper to run function-level optimizations (like LingoDB's pmFunc)
void createFunctionOptimizationPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("createFunctionOptimizationPipeline: Starting function-level optimizations");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    // Add function-level optimization passes
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createCSEPass());
    
    PGX_INFO("createFunctionOptimizationPipeline: Function optimizations configured");
}

} // namespace pgx_lower
} // namespace mlir