#include "mlir/Passes.h"
#include "execution/logging.h"

// Include all conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/DSAToLLVM/DSAToLLVM.h"

// Include dialect headers for verification
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Include MLIR infrastructure
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassInstrumentation.h"

// Include timing support
#include <chrono>
#include <sstream>

namespace mlir {
namespace pgx_lower {

//===----------------------------------------------------------------------===//
// Pass Timing Instrumentation
//===----------------------------------------------------------------------===//

class PgxPassTimingInstrumentation : public PassInstrumentation {
private:
    struct PassTiming {
        std::chrono::high_resolution_clock::time_point start;
        std::string passName;
    };
    
    std::vector<PassTiming> passStack;
    
public:
    void runBeforePass(Pass* pass, Operation* op) override {
        PassTiming timing;
        timing.start = std::chrono::high_resolution_clock::now();
        timing.passName = pass->getName().str();
        passStack.push_back(timing);
        
        PGX_DEBUG("Starting pass: " + timing.passName);
    }
    
    void runAfterPass(Pass* pass, Operation* op) override {
        if (passStack.empty()) return;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto& timing = passStack.back();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - timing.start);
        
        std::stringstream ss;
        ss << "Completed pass: " << timing.passName 
           << " in " << duration.count() << " microseconds";
        PGX_INFO(ss.str());
        
        passStack.pop_back();
    }
    
    void runAfterPassFailed(Pass* pass, Operation* op) override {
        if (passStack.empty()) return;
        
        auto& timing = passStack.back();
        PGX_ERROR("Failed pass: " + timing.passName);
        passStack.pop_back();
    }
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerAllPgxLoweringPasses() {
    PGX_DEBUG("Registering all pgx-lower conversion passes");
    
    // Register conversion passes in dependency order
    ::mlir::pgx_conversion::registerRelAlgToDBConversionPasses();
    ::mlir::pgx_conversion::registerDSAToLLVMConversionPasses();
    
    // TODO: Register future passes here
    // - Optimization passes
    // - Canonicalization passes
    
    PGX_INFO("All pgx-lower passes registered successfully");
}

//===----------------------------------------------------------------------===//
// Pipeline Configuration
//===----------------------------------------------------------------------===//

void createRelAlgToDBPipeline(mlir::PassManager& pm) {
    PGX_DEBUG("Creating RelAlg → DB lowering pipeline");
    
    // Phase 3a: RelAlg → DB lowering (nested since it's anchored on func::FuncOp)
    pm.addNestedPass<mlir::func::FuncOp>(::mlir::pgx_conversion::createRelAlgToDBPass());
    
    // Verification is handled by PassManager enableVerifier flag
    
    PGX_DEBUG("RelAlg → DB pipeline configured");
}


void createCompleteLoweringPipeline(mlir::PassManager& pm, bool enableVerifier) {
    auto pipelineStart = std::chrono::high_resolution_clock::now();
    PGX_INFO("Creating complete lowering pipeline for Test 1");
    
    // Enable verifier based on configuration
    pm.enableVerifier(enableVerifier);
    
    // Add timing instrumentation for performance monitoring
    pm.addInstrumentation(std::make_unique<PgxPassTimingInstrumentation>());
    PGX_DEBUG("Added pass timing instrumentation");
    
    // Since our passes are anchored on func::FuncOp, we need to nest them properly
    // Create a nested pass manager for function passes
    pm.addNestedPass<mlir::func::FuncOp>(::mlir::pgx_conversion::createRelAlgToDBPass());
    PGX_DEBUG("Added nested RelAlg → DB lowering pass");
    
    
    // Phase 4a - DSA → LLVM lowering (operates on module level)
    pm.addPass(::mlir::pgx_conversion::createDSAToLLVMPass());
    PGX_DEBUG("Added DSA → LLVM lowering pass");
    
    // TODO: Phase 4b - JIT execution engine setup
    // TODO: Phase 4c - Complete MLIR → LLVM → JIT pipeline
    
    // TODO: Optimization passes
    // pm.addPass(mlir::createCanonicalizerPass());
    // pm.addPass(mlir::createCSEPass());
    
    auto pipelineEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        pipelineEnd - pipelineStart);
    
    std::stringstream ss;
    ss << "Pipeline configuration completed in " << duration.count() << " microseconds";
    if (enableVerifier) {
        ss << " with verification enabled";
    } else {
        ss << " with verification disabled";
    }
    PGX_INFO(ss.str());
}

//===----------------------------------------------------------------------===//
// Future Pipeline Extensions
//===----------------------------------------------------------------------===//

// TODO: Implement DSA → LLVM pipeline for Phase 4a
// void createDSAToLLVMPipeline(mlir::PassManager& pm) {
//     PGX_DEBUG("Creating DSA → LLVM lowering pipeline");
//     // Phase 4a implementation
// }

// TODO: Implement optimization pipeline
// void createOptimizationPipeline(mlir::PassManager& pm) {
//     PGX_DEBUG("Creating optimization pipeline");
//     // Add optimization passes
// }

//===----------------------------------------------------------------------===//
// Library Validation Functions
//===----------------------------------------------------------------------===//

bool validateLibraryLoading() {
    PGX_DEBUG("Validating MLIR library loading");
    
    // Verify core MLIR symbols are available
    // Test by creating a temporary context
    mlir::MLIRContext testContext;
    
    // Verify pass manager can be created
    mlir::PassManager testPM(&testContext);
    
    // Verify instrumentation support
    testPM.addInstrumentation(std::make_unique<PgxPassTimingInstrumentation>());
    
    PGX_INFO("MLIR library loading validation passed");
    return true;
}

bool validateDialectRegistration() {
    PGX_DEBUG("Validating dialect registration");
    
    mlir::MLIRContext context;
    
    // Validate standard dialects
    auto funcDialect = context.getOrLoadDialect<mlir::func::FuncDialect>();
    if (!funcDialect) {
        PGX_ERROR("Failed to load Func dialect");
        return false;
    }
    
    auto arithDialect = context.getOrLoadDialect<mlir::arith::ArithDialect>();
    if (!arithDialect) {
        PGX_ERROR("Failed to load Arith dialect");
        return false;
    }
    
    // Validate custom dialects
    auto relalgDialect = context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
    if (!relalgDialect) {
        PGX_ERROR("Failed to load RelAlg dialect");
        return false;
    }
    
    auto dbDialect = context.getOrLoadDialect<pgx::db::DBDialect>();
    if (!dbDialect) {
        PGX_ERROR("Failed to load DB dialect");
        return false;
    }
    
    auto dsaDialect = context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
    if (!dsaDialect) {
        PGX_ERROR("Failed to load DSA dialect");
        return false;
    }
    
    PGX_INFO("All dialects successfully registered and loaded");
    return true;
}

bool validatePassRegistration() {
    PGX_DEBUG("Validating pass registration");
    
    // Verify passes can be created
    auto relalgToDBPass = ::mlir::pgx_conversion::createRelAlgToDBPass();
    if (!relalgToDBPass) {
        PGX_ERROR("Failed to create RelAlgToDB pass");
        return false;
    }
    
    
    auto dsaToLLVMPass = ::mlir::pgx_conversion::createDSAToLLVMPass();
    if (!dsaToLLVMPass) {
        PGX_ERROR("Failed to create DSAToLLVM pass");
        return false;
    }
    
    // Verify passes have valid names
    if (relalgToDBPass->getName().empty()) {
        PGX_ERROR("RelAlgToDB pass has empty name");
        return false;
    }
    
    
    if (dsaToLLVMPass->getName().empty()) {
        PGX_ERROR("DSAToLLVM pass has empty name");
        return false;
    }
    
    PGX_INFO("All passes successfully registered and can be instantiated");
    return true;
}

} // namespace pgx_lower
} // namespace mlir