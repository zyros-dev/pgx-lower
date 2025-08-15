#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"
#include "execution/jit_execution_engine.h"
#include "mlir/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include <memory>

namespace {

class SimpleJITDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create context with JIT-ready setup following LingoDB pattern
        context = std::make_shared<mlir::MLIRContext>();
        
        // Create registry with all required dialects and translation interfaces  
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                       mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                       mlir::cf::ControlFlowDialect>();
        
        // CRITICAL: Register translation interfaces for JIT execution
        mlir::registerAllToLLVMIRTranslations(registry);
        mlir::registerConvertFuncToLLVMInterface(registry);
        mlir::arith::registerConvertArithToLLVMInterface(registry);
        mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
        mlir::registerConvertMemRefToLLVMInterface(registry);
        
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
        
        // CRITICAL: Register base LLVM dialect translation in context (LingoDB pattern)
        mlir::registerLLVMDialectTranslation(*context);
    }
    
    // Helper to create simple MLIR module
    mlir::ModuleOp createSimpleModule() {
        mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
        mlir::OpBuilder builder(module.getBodyRegion());
        
        // Create "main" function returning i32
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "main", funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto constant = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 42, 32);
        builder.create<mlir::func::ReturnOp>(
            builder.getUnknownLoc(), constant.getResult());
        
        return module;
    }
    
    std::shared_ptr<mlir::MLIRContext> context;
};

// Test 1: Basic JIT Engine Creation and Initialization
TEST_F(SimpleJITDebugTest, TestJITEngineBasicCreation) {
    PGX_INFO("=== Testing basic JIT engine creation and initialization ===");
    
    // Create simple module
    mlir::ModuleOp module = createSimpleModule();
    PGX_INFO("✓ Created simple MLIR module");
    
    try {
        // Create JIT engine (correct API - no arguments to constructor)
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        PGX_INFO("✓ JIT engine created successfully");
        
        // Test initialization with module
        bool initSuccess = jitEngine->initialize(module);
        if (initSuccess) {
            PGX_INFO("✓ JIT engine initialization succeeded");
            EXPECT_TRUE(jitEngine->isInitialized()) << "Engine should report as initialized";
        } else {
            PGX_ERROR("✗ JIT engine initialization failed");
            FAIL() << "JIT initialization failed - this explains PostgreSQL crashes";
        }
        
        // Test LLVM compilation
        bool compileSuccess = jitEngine->compileToLLVMIR(module);
        if (compileSuccess) {
            PGX_INFO("✓ LLVM compilation succeeded");
            PGX_INFO("✓ Complete JIT pipeline works in pure C++ environment");
        } else {
            PGX_ERROR("✗ LLVM compilation failed");
            PGX_ERROR("This indicates the JIT compilation step is broken");
            FAIL() << "LLVM compilation failed - root cause of PostgreSQL crashes";
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("✗ Exception during JIT engine test: " + std::string(e.what()));
        FAIL() << "JIT engine threw exception: " << e.what();
    }
}

// Test 2: JIT Engine API Method Testing  
TEST_F(SimpleJITDebugTest, TestJITEngineAPIMethods) {
    PGX_INFO("=== Testing JIT engine API methods ===");
    
    mlir::ModuleOp module = createSimpleModule();
    
    try {
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        
        // Test optimization level setting
        jitEngine->setOptimizationLevel(llvm::CodeGenOptLevel::None);
        PGX_INFO("✓ Optimization level setting works");
        
        // Test initialization state before initialize()
        EXPECT_FALSE(jitEngine->isInitialized()) << "Engine should not be initialized initially";
        
        // Test initialization
        bool initSuccess = jitEngine->initialize(module);
        EXPECT_TRUE(initSuccess) << "Initialization should succeed";
        
        if (initSuccess) {
            // Test initialization state after initialize()
            EXPECT_TRUE(jitEngine->isInitialized()) << "Engine should be initialized after initialize()";
            
            // Test setup methods
            bool setupSuccess = jitEngine->setupJITOptimizationPipeline();
            if (setupSuccess) {
                PGX_INFO("✓ JIT optimization pipeline setup succeeded");
            } else {
                PGX_WARNING("⚠ JIT optimization pipeline setup failed");
            }
            
            // Test memory context setup (without actual PostgreSQL)
            bool memorySuccess = jitEngine->setupMemoryContexts();
            if (memorySuccess) {
                PGX_INFO("✓ Memory context setup succeeded");
            } else {
                PGX_WARNING("⚠ Memory context setup failed (expected without PostgreSQL)");
            }
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("✗ Exception during API method testing: " + std::string(e.what()));
        FAIL() << "JIT API method testing failed: " << e.what();
    }
}

// Test 3: Multiple Module Compilation Test
TEST_F(SimpleJITDebugTest, TestMultipleModuleCompilation) {
    PGX_INFO("=== Testing multiple module compilation ===");
    
    const int num_modules = 5;
    int successful_compilations = 0;
    
    for (int i = 0; i < num_modules; i++) {
        try {
            mlir::ModuleOp module = createSimpleModule();
            auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
            
            bool initSuccess = jitEngine->initialize(module);
            if (initSuccess) {
                bool compileSuccess = jitEngine->compileToLLVMIR(module);
                if (compileSuccess) {
                    successful_compilations++;
                    PGX_INFO("✓ Module " + std::to_string(i + 1) + " compiled successfully");
                } else {
                    PGX_ERROR("✗ Module " + std::to_string(i + 1) + " compilation failed");
                }
            } else {
                PGX_ERROR("✗ Module " + std::to_string(i + 1) + " initialization failed");
            }
            
        } catch (const std::exception& e) {
            PGX_ERROR("✗ Module " + std::to_string(i + 1) + " threw exception: " + std::string(e.what()));
        }
    }
    
    double success_rate = (double)successful_compilations / num_modules;
    PGX_INFO("JIT compilation success rate: " + std::to_string(success_rate * 100) + "% (" + 
            std::to_string(successful_compilations) + "/" + std::to_string(num_modules) + ")");
    
    if (success_rate > 0.8) {
        PGX_INFO("✓ JIT engine shows good reliability");
        SUCCEED() << "High JIT compilation success rate";
    } else {
        PGX_ERROR("✗ JIT engine shows poor reliability");
        FAIL() << "Low JIT compilation success rate: " << (success_rate * 100) << "%";
    }
}

// Test 4: Error Handling and Recovery Test
TEST_F(SimpleJITDebugTest, TestJITEngineErrorHandling) {
    PGX_INFO("=== Testing JIT engine error handling and recovery ===");
    
    try {
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        
        // Test behavior with invalid module (null)
        mlir::ModuleOp invalidModule;  // Default constructed - should be invalid
        bool initWithInvalid = jitEngine->initialize(invalidModule);
        
        if (!initWithInvalid) {
            PGX_INFO("✓ JIT engine correctly rejects invalid module");
        } else {
            PGX_WARNING("⚠ JIT engine accepted invalid module - potential issue");
        }
        
        // Test with valid module after invalid attempt
        mlir::ModuleOp validModule = createSimpleModule();
        bool initWithValid = jitEngine->initialize(validModule);
        
        if (initWithValid) {
            PGX_INFO("✓ JIT engine recovered and accepts valid module");
            
            // Test compilation after recovery
            bool compileAfterRecovery = jitEngine->compileToLLVMIR(validModule);
            if (compileAfterRecovery) {
                PGX_INFO("✓ JIT compilation works after error recovery");
            } else {
                PGX_ERROR("✗ JIT compilation failed after error recovery");
            }
        } else {
            PGX_ERROR("✗ JIT engine failed to accept valid module after invalid attempt");
            FAIL() << "JIT engine error recovery failed";
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("✗ Exception during error handling test: " + std::string(e.what()));
        FAIL() << "JIT error handling test failed: " << e.what();
    }
}

} // namespace