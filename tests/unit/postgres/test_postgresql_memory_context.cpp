#include <gtest/gtest.h>

// PostgreSQL headers for memory management
extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "miscadmin.h"
#include "storage/ipc.h"
#include "utils/elog.h"
}

// Our headers
#include "execution/logging.h"
#include "execution/jit_execution_engine.h"

// MLIR headers for test modules
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "compiler/Pipeline/StandardToLLVMPipeline.h"

namespace {

class PostgreSQLMemoryContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_INFO("=== Setting up PostgreSQL memory context test environment ===");
        
        // Initialize PostgreSQL memory management subsystem
        // This creates the basic memory context hierarchy without database connection
        try {
            // Create top-level memory context (like PostgreSQL main() does)
            TopMemoryContext = AllocSetContextCreate(NULL,
                                                     "TopMemoryContext",
                                                     ALLOCSET_DEFAULT_SIZES);
            
            // Create current memory context for allocations
            CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
                                                        "CurrentMemoryContext",
                                                        ALLOCSET_DEFAULT_SIZES);
            
            // Create error context for exception handling
            ErrorContext = AllocSetContextCreate(TopMemoryContext,
                                               "ErrorContext",
                                               ALLOCSET_DEFAULT_SIZES);
            
            PGX_INFO("✓ PostgreSQL memory context hierarchy initialized");
            memory_contexts_initialized = true;
            
        } catch (...) {
            PGX_ERROR("✗ Failed to initialize PostgreSQL memory contexts");
            memory_contexts_initialized = false;
        }
        
        // Initialize MLIR context
        mlir_context = std::make_shared<mlir::MLIRContext>();
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
        mlir_context->appendDialectRegistry(registry);
        mlir_context->loadAllAvailableDialects();
    }
    
    void TearDown() override {
        if (memory_contexts_initialized) {
            // Clean up PostgreSQL memory contexts in reverse order
            if (ErrorContext) {
                MemoryContextDelete(ErrorContext);
                ErrorContext = nullptr;
            }
            if (CurrentMemoryContext && CurrentMemoryContext != TopMemoryContext) {
                MemoryContextDelete(CurrentMemoryContext);
                CurrentMemoryContext = nullptr;
            }
            if (TopMemoryContext) {
                MemoryContextDelete(TopMemoryContext);
                TopMemoryContext = nullptr;
            }
        }
    }
    
    // Helper to create test MLIR module
    mlir::ModuleOp createTestModule() {
        mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(mlir_context.get()));
        mlir::OpBuilder builder(module.getBodyRegion());
        
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "main", funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto constant = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 42, 32);
        builder.create<mlir::func::ReturnOp>(
            builder.getUnknownLoc(), constant.getResult());
        
        // Apply Standard→LLVM pipeline
        mlir::PassManager pm(mlir_context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        auto result = pm.run(module);
        assert(mlir::succeeded(result) && "MLIR lowering must succeed in test");
        
        return module;
    }
    
    bool memory_contexts_initialized = false;
    std::shared_ptr<mlir::MLIRContext> mlir_context;
    MemoryContext TopMemoryContext = nullptr;
    MemoryContext CurrentMemoryContext = nullptr;
    MemoryContext ErrorContext = nullptr;
};

// Test 1: Basic Memory Context Operations with JIT
TEST_F(PostgreSQLMemoryContextTest, TestBasicMemoryContextOperations) {
    ASSERT_TRUE(memory_contexts_initialized) << "PostgreSQL memory contexts must be initialized";
    
    PGX_INFO("=== Testing basic memory context operations with JIT execution ===");
    
    // Create a child memory context for this test
    MemoryContext testContext = AllocSetContextCreate(CurrentMemoryContext,
                                                     "TestContext",
                                                     ALLOCSET_DEFAULT_SIZES);
    
    // Switch to test context for all allocations
    MemoryContext oldContext = MemoryContextSwitchTo(testContext);
    
    PGX_INFO("✓ Created and switched to TestContext");
    
    // Create MLIR module within test context
    mlir::ModuleOp module = createTestModule();
    PGX_INFO("✓ Created MLIR module in TestContext");
    
    // Test JIT execution in test context
    bool jit_in_test_context = false;
    try {
        auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
        bool initSuccess = jitEngine->initialize();
        if (initSuccess) {
            bool lookupSuccess = jitEngine->lookupCompiledQuery();
            jit_in_test_context = lookupSuccess;
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Exception in TestContext: " + std::string(e.what()));
    }
    
    if (jit_in_test_context) {
        PGX_INFO("✓ JIT execution succeeded in TestContext");
    } else {
        PGX_ERROR("✗ JIT execution failed in TestContext");
    }
    
    // Test context switching behavior
    MemoryContextSwitchTo(CurrentMemoryContext);
    PGX_INFO("Switched back to CurrentMemoryContext");
    
    // Test if JIT still works after context switch
    bool jit_after_switch = false;
    try {
        auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
        bool initSuccess = jitEngine->initialize();
        if (initSuccess) {
            bool lookupSuccess = jitEngine->lookupCompiledQuery();
            jit_after_switch = lookupSuccess;
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Exception after context switch: " + std::string(e.what()));
    }
    
    if (jit_after_switch) {
        PGX_INFO("✓ JIT execution survived context switch");
    } else {
        PGX_ERROR("✗ JIT execution failed after context switch");
    }
    
    // Restore original context and clean up
    MemoryContextSwitchTo(oldContext);
    MemoryContextDelete(testContext);
    
    // Report results
    EXPECT_TRUE(jit_in_test_context) << "JIT should work in test context";
    EXPECT_TRUE(jit_after_switch) << "JIT should survive context switches";
}

// Test 2: Memory Context Reset Effects on JIT
TEST_F(PostgreSQLMemoryContextTest, TestMemoryContextResetEffects) {
    ASSERT_TRUE(memory_contexts_initialized) << "PostgreSQL memory contexts must be initialized";
    
    PGX_INFO("=== Testing memory context reset effects on JIT execution ===");
    
    // Create query context (simulates PostgreSQL query execution environment)
    MemoryContext queryContext = AllocSetContextCreate(CurrentMemoryContext,
                                                       "QueryContext",
                                                       ALLOCSET_DEFAULT_SIZES);
    
    MemoryContext oldContext = MemoryContextSwitchTo(queryContext);
    
    // Create MLIR module and JIT in query context
    mlir::ModuleOp module = createTestModule();
    PGX_INFO("✓ Created MLIR module in QueryContext");
    
    // Test initial JIT execution
    bool initial_jit_success = false;
    std::unique_ptr<::pgx_lower::JITExecutionEngine> persistentEngine;
    
    try {
        persistentEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
        bool initSuccess = persistentEngine->initialize();
        if (initSuccess) {
            bool lookupSuccess = persistentEngine->lookupCompiledQuery();
            initial_jit_success = lookupSuccess;
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Initial JIT execution failed: " + std::string(e.what()));
    }
    
    if (initial_jit_success) {
        PGX_INFO("✓ Initial JIT execution succeeded");
    } else {
        PGX_ERROR("✗ Initial JIT execution failed");
        MemoryContextSwitchTo(oldContext);
        MemoryContextDelete(queryContext);
        FAIL() << "Cannot test reset effects without working initial JIT";
        return;
    }
    
    // Test 1: Soft reset (clear allocations but keep context)
    PGX_INFO("Testing soft reset (MemoryContextReset)");
    MemoryContextReset(queryContext);
    
    bool jit_after_soft_reset = false;
    try {
        // Test if existing JIT engine still works
        if (persistentEngine) {
            bool lookupSuccess = persistentEngine->lookupCompiledQuery();
            jit_after_soft_reset = lookupSuccess;
        }
    } catch (const std::exception& e) {
        PGX_ERROR("JIT failed after soft reset: " + std::string(e.what()));
    }
    
    if (jit_after_soft_reset) {
        PGX_INFO("✓ JIT execution survived soft reset");
    } else {
        PGX_ERROR("✗ JIT execution failed after soft reset");
        PGX_ERROR("This indicates memory invalidation is breaking LLVM JIT");
    }
    
    // Test 2: Create new JIT engine after reset
    bool new_jit_after_reset = false;
    try {
        // Module might be invalidated, create new one
        mlir::ModuleOp newModule = createTestModule();
        auto newJitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(newModule);
        bool initSuccess = newJitEngine->initialize();
        if (initSuccess) {
            bool lookupSuccess = newJitEngine->lookupCompiledQuery();
            new_jit_after_reset = lookupSuccess;
        }
    } catch (const std::exception& e) {
        PGX_ERROR("New JIT creation failed after reset: " + std::string(e.what()));
    }
    
    if (new_jit_after_reset) {
        PGX_INFO("✓ New JIT creation succeeded after reset");
    } else {
        PGX_ERROR("✗ New JIT creation failed after reset");
    }
    
    // Test 3: Simulate PostgreSQL LOAD command scenario
    PGX_INFO("Simulating PostgreSQL LOAD command memory invalidation");
    
    // Switch to different context before invalidation
    MemoryContextSwitchTo(CurrentMemoryContext);
    
    // Reset the query context (simulates LOAD command behavior)
    MemoryContextReset(queryContext);
    
    // Try to access JIT from different context
    bool jit_cross_context = false;
    try {
        mlir::ModuleOp crossModule = createTestModule();
        auto crossJitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(crossModule);
        bool initSuccess = crossJitEngine->initialize();
        if (initSuccess) {
            bool lookupSuccess = crossJitEngine->lookupCompiledQuery();
            jit_cross_context = lookupSuccess;
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Cross-context JIT failed: " + std::string(e.what()));
    }
    
    if (jit_cross_context) {
        PGX_INFO("✓ JIT works across context boundaries after reset");
    } else {
        PGX_ERROR("✗ JIT fails across context boundaries after reset");
        PGX_ERROR("This reproduces the PostgreSQL LOAD command crash pattern");
    }
    
    // Restore context and clean up
    MemoryContextSwitchTo(oldContext);
    MemoryContextDelete(queryContext);
    
    // Analyze results
    if (!jit_after_soft_reset && !new_jit_after_reset) {
        PGX_ERROR("=== CRITICAL FINDING ===");
        PGX_ERROR("Memory context resets completely break JIT execution");
        PGX_ERROR("This explains PostgreSQL LOAD command crashes");
        FAIL() << "Memory context invalidation breaks JIT - root cause identified";
    } else if (!jit_after_soft_reset && new_jit_after_reset) {
        PGX_INFO("=== IMPORTANT FINDING ===");
        PGX_INFO("Existing JIT engines break on reset, but new ones work");
        PGX_INFO("Solution: Recreate JIT engines after PostgreSQL LOAD commands");
        SUCCEED() << "Partial memory context incompatibility identified";
    } else {
        PGX_INFO("✓ JIT execution is resilient to memory context operations");
        SUCCEED() << "Memory contexts not the root cause";
    }
}

// Test 3: Long-running Memory Context Stress Test
TEST_F(PostgreSQLMemoryContextTest, TestLongRunningMemoryContextStress) {
    ASSERT_TRUE(memory_contexts_initialized) << "PostgreSQL memory contexts must be initialized";
    
    PGX_INFO("=== Testing JIT resilience under memory context stress ===");
    
    const int num_iterations = 50;
    int successful_iterations = 0;
    
    for (int i = 0; i < num_iterations; i++) {
        // Create new context for each iteration
        std::string contextName = "StressContext" + std::to_string(i);
        MemoryContext stressContext = AllocSetContextCreate(CurrentMemoryContext,
                                                           contextName.c_str(),
                                                           ALLOCSET_DEFAULT_SIZES);
        
        MemoryContext oldContext = MemoryContextSwitchTo(stressContext);
        
        bool iteration_success = false;
        try {
            // Create MLIR module in this context
            mlir::ModuleOp module = createTestModule();
            
            // Test JIT execution
            auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
            bool initSuccess = jitEngine->initialize();
            if (initSuccess) {
                bool lookupSuccess = jitEngine->lookupCompiledQuery();
                iteration_success = lookupSuccess;
            }
            
        } catch (const std::exception& e) {
            PGX_DEBUG("Iteration " + std::to_string(i) + " failed: " + std::string(e.what()));
        }
        
        if (iteration_success) {
            successful_iterations++;
        }
        
        // Clean up context
        MemoryContextSwitchTo(oldContext);
        MemoryContextDelete(stressContext);
        
        // Periodic status report
        if ((i + 1) % 10 == 0) {
            PGX_INFO("Completed " + std::to_string(i + 1) + "/" + std::to_string(num_iterations) + 
                    " iterations, success rate: " + 
                    std::to_string((successful_iterations * 100) / (i + 1)) + "%");
        }
    }
    
    double success_rate = (double)successful_iterations / num_iterations;
    PGX_INFO("Final success rate: " + std::to_string(success_rate * 100) + "% (" + 
            std::to_string(successful_iterations) + "/" + std::to_string(num_iterations) + ")");
    
    if (success_rate > 0.95) {
        PGX_INFO("✓ JIT execution is highly stable under memory context stress");
        SUCCEED() << "High stability under memory context operations";
    } else if (success_rate > 0.80) {
        PGX_WARNING("⚠ JIT execution shows occasional failures under memory context stress");
        PGX_WARNING("This suggests intermittent memory context compatibility issues");
        SUCCEED() << "Moderate stability - intermittent issues detected";
    } else {
        PGX_ERROR("✗ JIT execution frequently fails under memory context stress");
        PGX_ERROR("This indicates serious memory context compatibility problems");
        FAIL() << "Poor stability under memory context stress: " << (success_rate * 100) << "%";
    }
}

} // namespace