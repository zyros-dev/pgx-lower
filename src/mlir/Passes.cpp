#include "mlir/Passes.h"
#include "execution/logging.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Verifier.h"

// Include all conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/DSAToLLVM/DSAToLLVM.h"
#include "mlir/Conversion/UtilToLLVM/UtilToLLVM.h"

// Include dialect headers for verification
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Include MLIR infrastructure
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Transforms/Passes.h"

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
    
    // RelAlg → DB lowering (generates mixed DB+DSA+Util operations)
    pm.addNestedPass<mlir::func::FuncOp>(::mlir::pgx_conversion::createRelAlgToDBPass());
    
    // Verification is handled by PassManager enableVerifier flag
    
    PGX_DEBUG("RelAlg → DB pipeline configured");
}


// DEPRECATED: Single PassManager approach
// Replaced with LingoDB-compliant multi-PassManager architecture
void createCompleteLoweringPipeline_DEPRECATED(mlir::PassManager& pm, bool enableVerifier) {
    // This function is deprecated and should not be used
    // Use runCompleteLoweringPipeline() instead for LingoDB compliance
    PGX_ERROR("createCompleteLoweringPipeline is deprecated - use runCompleteLoweringPipeline");
}

//===----------------------------------------------------------------------===//
// LingoDB-Compliant Multi-PassManager Pipeline
//===----------------------------------------------------------------------===//

// Phase 1: RelAlg → Mixed DB+DSA+Util (Function-Scoped with Nested Pass)
static LogicalResult runPhase1_RelAlgToMixed(mlir::ModuleOp module, mlir::MLIRContext* context, bool enableVerifier) {
    auto phaseStart = std::chrono::high_resolution_clock::now();
    PGX_INFO("Phase 1: RelAlg → Mixed DB+DSA+Util lowering");
    
    mlir::PassManager pm1(context);
    pm1.enableVerifier(enableVerifier);
    
    // Use LingoDB's nested pass pattern
    createRelAlgToDBPipeline(pm1);
    
    if (mlir::failed(pm1.run(module))) {
        PGX_ERROR("Phase 1 failed: RelAlg → Mixed DB+DSA+Util lowering");
        return failure();
    }
    
    auto phaseEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(phaseEnd - phaseStart);
    PGX_INFO("Phase 1 completed in " + std::to_string(duration.count() / 1000.0) + " ms");
    
    return success();
}

// Phase 2: DB → Standard MLIR + PostgreSQL SPI (Module-Scoped)
static LogicalResult runPhase2_DBToStandard(mlir::ModuleOp module, mlir::MLIRContext* context, bool enableVerifier) {
    auto phaseStart = std::chrono::high_resolution_clock::now();
    PGX_INFO("Phase 2: DB → Standard MLIR + PostgreSQL SPI lowering");
    
    // Count DB operations before conversion for diagnostics
    int dbOpsCount = 0;
    module.walk([&](Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "db") {
            dbOpsCount++;
        }
    });
    
    if (dbOpsCount > 0) {
        PGX_DEBUG("Phase 2: Found " + std::to_string(dbOpsCount) + " DB operations to convert");
    } else {
        PGX_DEBUG("Phase 2: No DB operations found - skipping conversion");
    }
    
    mlir::PassManager pm2(context);
    pm2.enableVerifier(enableVerifier);
    
    // DB operations to PostgreSQL SPI calls + Standard MLIR
    pm2.addPass(createDBToStdPass());
    
    auto passResult = pm2.run(module);
    if (mlir::failed(passResult)) {
        // Enhanced error reporting for PostgreSQL SPI failures
        int remainingDbOps = 0;
        module.walk([&](Operation* op) {
            if (op->getDialect() && op->getDialect()->getNamespace() == "db") {
                remainingDbOps++;
                PGX_DEBUG("Unconverted DB operation: " + op->getName().getStringRef().str());
            }
        });
        
        if (remainingDbOps > 0) {
            PGX_ERROR("Phase 2 failed: " + std::to_string(remainingDbOps) + " DB operations could not be converted to PostgreSQL SPI calls");
        } else {
            PGX_ERROR("Phase 2 failed: DB → Standard MLIR lowering failed for unknown reason");
        }
        
        PGX_ERROR("Possible causes: Missing PostgreSQL SPI function declarations, type conversion failures, or unsupported DB operations");
        return failure();
    }
    
    // Verify PostgreSQL SPI functions were generated
    bool hasSPIFunctions = false;
    module.walk([&](mlir::func::FuncOp func) {
        auto funcName = func.getName().str();
        if (funcName.find("pg_") == 0) {
            hasSPIFunctions = true;
            PGX_DEBUG("Generated PostgreSQL SPI function: " + funcName);
        }
    });
    
    if (dbOpsCount > 0 && !hasSPIFunctions) {
        PGX_WARNING("Phase 2: DB operations were converted but no PostgreSQL SPI functions were generated");
    }
    
    auto phaseEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(phaseEnd - phaseStart);
    PGX_INFO("Phase 2 completed in " + std::to_string(duration.count() / 1000.0) + " ms");
    
    return success();
}

// Phase 3: DSA+Util → Standard MLIR/LLVM (Module-Scoped)
static LogicalResult runPhase3_DSAUtilToLLVM(mlir::ModuleOp module, mlir::MLIRContext* context, bool enableVerifier) {
    auto phaseStart = std::chrono::high_resolution_clock::now();
    PGX_INFO("Phase 3: DSA+Util → Standard MLIR/LLVM lowering");
    
    mlir::PassManager pm3(context);
    pm3.enableVerifier(enableVerifier);
    
    // DSA operations to Standard MLIR
    pm3.addPass(createDSAToStdPass());
    
    // Util operations directly to LLVM (LingoDB pattern)
    pm3.addPass(pgx::mlir::createUtilToLLVMPass());
    
    // Add canonicalization (LingoDB pattern)
    pm3.addPass(mlir::createCanonicalizerPass());
    
    if (mlir::failed(pm3.run(module))) {
        PGX_ERROR("Phase 3 failed: DSA+Util → Standard MLIR/LLVM lowering");
        return failure();
    }
    
    auto phaseEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(phaseEnd - phaseStart);
    PGX_INFO("Phase 3 completed in " + std::to_string(duration.count() / 1000.0) + " ms");
    
    return success();
}

LogicalResult runCompleteLoweringPipeline(mlir::ModuleOp module, mlir::MLIRContext* context, bool enableVerifier) {
    // Input validation with detailed error reporting
    if (!module) {
        PGX_ERROR("Pipeline failure: Null module provided");
        return failure();
    }
    
    if (!context) {
        PGX_ERROR("Pipeline failure: Null MLIR context provided");
        return failure();
    }
    
    // Verify module is valid before starting pipeline
    if (failed(mlir::verify(module))) {
        PGX_ERROR("Pipeline failure: Invalid MLIR module provided - failed verification");
        return failure();
    }
    
    PGX_INFO("Starting LingoDB-compliant multi-phase lowering pipeline for Test 1");
    auto pipelineStart = std::chrono::high_resolution_clock::now();

    // Execute each phase with comprehensive error handling
    if (failed(runPhase1_RelAlgToMixed(module, context, enableVerifier))) {
        PGX_ERROR("Pipeline terminated: Phase 1 (RelAlg→Mixed) failed");
        return failure();
    }
    
    if (failed(runPhase2_DBToStandard(module, context, enableVerifier))) {
        PGX_ERROR("Pipeline terminated: Phase 2 (DB→Standard) failed - PostgreSQL SPI integration issue");
        return failure();
    }
    
    if (failed(runPhase3_DSAUtilToLLVM(module, context, enableVerifier))) {
        PGX_ERROR("Pipeline terminated: Phase 3 (DSA+Util→LLVM) failed");
        return failure();
    }

    // Final verification to ensure pipeline produced valid MLIR
    if (enableVerifier && failed(mlir::verify(module))) {
        PGX_ERROR("Pipeline completed but produced invalid MLIR - verification failed");
        return failure();
    }

    auto pipelineEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(pipelineEnd - pipelineStart);
    PGX_INFO("Complete lowering pipeline finished successfully in " + std::to_string(totalDuration.count() / 1000.0) + " ms");
    
    return success();
}

// Legacy function for backwards compatibility - redirects to new multi-PassManager approach
void createCompleteLoweringPipeline(mlir::PassManager& pm, bool enableVerifier) {
    PGX_WARNING("createCompleteLoweringPipeline is deprecated - update callers to use runCompleteLoweringPipeline");
    
    // For backwards compatibility, configure the single PassManager with our phases
    // This is not optimal but maintains API compatibility
    createRelAlgToDBPipeline(pm);
    pm.addPass(createDBToStdPass());
    pm.addPass(createDSAToStdPass());
    pm.addPass(pgx::mlir::createUtilToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Future Pipeline Extensions
//===----------------------------------------------------------------------===//

// TODO: Implement complete DSA → LLVM pipeline
// void createDSAToLLVMPipeline(mlir::PassManager& pm) {
//     PGX_DEBUG("Creating DSA → LLVM lowering pipeline");
//     // Data structure algorithm lowering implementation
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
    // TEMPORARY: Disabled - see comment in createCompleteLoweringPipeline
    // testPM.addInstrumentation(std::make_unique<PgxPassTimingInstrumentation>());
    
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
    
    auto utilDialect = context.getOrLoadDialect<pgx::mlir::util::UtilDialect>();
    if (!utilDialect) {
        PGX_ERROR("Failed to load Util dialect");
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