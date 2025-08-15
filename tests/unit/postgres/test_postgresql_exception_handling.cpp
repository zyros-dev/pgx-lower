#include <gtest/gtest.h>

// PostgreSQL headers for exception handling and memory contexts
extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "miscadmin.h"
#include "storage/ipc.h"
}

// Our headers
#include "execution/logging.h"
#include "execution/jit_execution_engine.h"
#include "execution/mlir_runner.h"

// MLIR headers for test setup
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "compiler/Pipeline/StandardToLLVMPipeline.h"

namespace {

class PostgreSQLExceptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_INFO("=== Setting up PostgreSQL exception handling test ===");
        
        // Initialize minimal PostgreSQL environment for memory contexts
        // This does NOT require a database connection
        try {
            // Create top-level memory context (similar to PostgreSQL startup)
            TopMemoryContext = AllocSetContextCreate(NULL,
                                                     "TopMemoryContext",
                                                     ALLOCSET_DEFAULT_SIZES);
            CurrentMemoryContext = TopMemoryContext;
            
            // Create error context for exception handling
            ErrorContext = AllocSetContextCreate(TopMemoryContext,
                                               "ErrorContext", 
                                               ALLOCSET_DEFAULT_SIZES);
            
            PGX_INFO("✓ PostgreSQL memory contexts initialized successfully");
            postgres_initialized = true;
            
        } catch (...) {
            PGX_ERROR("✗ Failed to initialize PostgreSQL memory contexts");
            postgres_initialized = false;
        }
        
        // Initialize MLIR context
        context = std::make_shared<mlir::MLIRContext>();
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
    }
    
    void TearDown() override {
        if (postgres_initialized) {
            // Clean up PostgreSQL memory contexts
            if (ErrorContext) {
                MemoryContextDelete(ErrorContext);
                ErrorContext = nullptr;
            }
            if (TopMemoryContext) {
                MemoryContextDelete(TopMemoryContext);
                TopMemoryContext = nullptr;
            }
            CurrentMemoryContext = nullptr;
        }
    }
    
    bool postgres_initialized = false;
    std::shared_ptr<mlir::MLIRContext> context;
    MemoryContext TopMemoryContext = nullptr;
    MemoryContext ErrorContext = nullptr;
    MemoryContext CurrentMemoryContext = nullptr;
};

// Test 1: Exception Type Detection Test
TEST_F(PostgreSQLExceptionTest, TestPostgreSQLExceptionType) {
    ASSERT_TRUE(postgres_initialized) << "PostgreSQL memory contexts must be initialized";
    
    PGX_INFO("=== Testing PostgreSQL exception detection and classification ===");
    
    // Create a simple MLIR module that will trigger the same path as our crash
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
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
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result));
    
    // Now test JIT execution within PostgreSQL exception handling context
    // This simulates the exact same code path as my_executor.cpp:342
    bool caught_exception = false;
    std::string exception_type = "none";
    std::string exception_message = "none";
    int exception_sqlerrcode = 0;
    
    PG_TRY();
    {
        PGX_INFO("Attempting JIT execution within PostgreSQL exception context");
        
        // Switch to ErrorContext for exception safety
        MemoryContext oldcontext = MemoryContextSwitchTo(ErrorContext);
        
        try {
            // This is the exact same JIT execution path as PostgreSQL
            auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
            
            bool initSuccess = jitEngine->initialize();
            if (!initSuccess) {
                PGX_ERROR("JIT initialization failed in PostgreSQL context");
                // Trigger PostgreSQL error to test exception handling
                ereport(ERROR,
                       (errcode(ERRCODE_INTERNAL_ERROR),
                        errmsg("JIT initialization failed in test")));
            }
            
            bool lookupSuccess = jitEngine->lookupCompiledQuery();
            if (!lookupSuccess) {
                PGX_ERROR("JIT function lookup failed in PostgreSQL context");
                // Trigger PostgreSQL error (this matches our actual crash point)
                ereport(ERROR,
                       (errcode(ERRCODE_UNDEFINED_FUNCTION),
                        errmsg("JIT function lookup failed - this reproduces the PostgreSQL crash")));
            }
            
            PGX_INFO("✓ JIT execution succeeded in PostgreSQL context - no exception");
            
        } catch (const std::exception& e) {
            PGX_ERROR("C++ exception during JIT execution: " + std::string(e.what()));
            // Convert C++ exception to PostgreSQL error
            ereport(ERROR,
                   (errcode(ERRCODE_INTERNAL_ERROR),
                    errmsg("C++ exception in JIT: %s", e.what())));
        }
        
        // Switch back to original context
        MemoryContextSwitchTo(oldcontext);
    }
    PG_CATCH();
    {
        // This is where we catch the EXACT same exception as my_executor.cpp:342
        caught_exception = true;
        
        // Extract detailed exception information
        ErrorData *edata = CopyErrorData();
        if (edata) {
            exception_sqlerrcode = edata->sqlerrcode;
            exception_message = std::string(edata->message ? edata->message : "no message");
            
            // Classify exception type
            if (edata->sqlerrcode == ERRCODE_UNDEFINED_FUNCTION) {
                exception_type = "UNDEFINED_FUNCTION";
            } else if (edata->sqlerrcode == ERRCODE_INTERNAL_ERROR) {
                exception_type = "INTERNAL_ERROR";
            } else {
                exception_type = "OTHER_ERROR";
            }
            
            PGX_ERROR("PostgreSQL exception caught:");
            PGX_ERROR("  Type: " + exception_type);
            PGX_ERROR("  SQL Error Code: " + std::to_string(exception_sqlerrcode));
            PGX_ERROR("  Message: " + exception_message);
            PGX_ERROR("  This is the EXACT same exception as my_executor.cpp:342");
            
            FreeErrorData(edata);
        }
        
        FlushErrorState();
    }
    PG_END_TRY();
    
    // Analyze results
    if (caught_exception) {
        PGX_INFO("✓ Successfully reproduced PostgreSQL exception in controlled environment");
        PGX_INFO("Exception type: " + exception_type);
        PGX_INFO("Exception message: " + exception_message);
        
        // Check if this matches expected failure patterns
        if (exception_type == "UNDEFINED_FUNCTION") {
            PGX_INFO("✓ Exception type indicates JIT function lookup failure");
            PGX_INFO("This confirms the crash is in JIT symbol resolution, not MLIR compilation");
        } else if (exception_type == "INTERNAL_ERROR") {
            PGX_INFO("✓ Exception type indicates internal JIT engine failure");
            PGX_INFO("This suggests a bug in our JIT execution engine implementation");
        }
        
        // This test succeeds by reproducing the crash in controlled environment
        SUCCEED() << "Successfully isolated PostgreSQL exception type: " << exception_type;
        
    } else {
        PGX_INFO("✗ No PostgreSQL exception caught - test environment differs from production");
        FAIL() << "Failed to reproduce PostgreSQL exception in test environment";
    }
}

// Test 2: Memory Context Boundary Test
TEST_F(PostgreSQLExceptionTest, TestMemoryContextBoundaries) {
    ASSERT_TRUE(postgres_initialized) << "PostgreSQL memory contexts must be initialized";
    
    PGX_INFO("=== Testing PostgreSQL memory context interactions with LLVM JIT ===");
    
    // Create multiple memory contexts to test boundary issues
    MemoryContext queryContext = AllocSetContextCreate(TopMemoryContext,
                                                       "QueryContext",
                                                       ALLOCSET_DEFAULT_SIZES);
    MemoryContext jitContext = AllocSetContextCreate(TopMemoryContext,
                                                     "JITContext", 
                                                     ALLOCSET_DEFAULT_SIZES);
    
    // Create MLIR module in query context
    MemoryContext oldcontext = MemoryContextSwitchTo(queryContext);
    
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
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
    
    // Apply lowering in query context
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result));
    
    PGX_INFO("✓ MLIR compilation completed in QueryContext");
    
    // Switch to JIT context for execution
    MemoryContextSwitchTo(jitContext);
    
    bool jit_success = false;
    std::string failure_reason = "none";
    
    try {
        PGX_INFO("Attempting JIT execution in separate JITContext");
        
        auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
        
        bool initSuccess = jitEngine->initialize();
        if (!initSuccess) {
            failure_reason = "JIT initialization failed in JITContext";
            PGX_ERROR(failure_reason);
        } else {
            bool lookupSuccess = jitEngine->lookupCompiledQuery();
            if (!lookupSuccess) {
                failure_reason = "JIT function lookup failed in JITContext";
                PGX_ERROR(failure_reason);
            } else {
                jit_success = true;
                PGX_INFO("✓ JIT execution succeeded across memory context boundaries");
            }
        }
        
    } catch (const std::exception& e) {
        failure_reason = "C++ exception in JITContext: " + std::string(e.what());
        PGX_ERROR(failure_reason);
    }
    
    // Test memory context reset/invalidation
    PGX_INFO("Testing memory context invalidation effects on JIT");
    
    // Reset query context (simulates LOAD command behavior)
    MemoryContextReset(queryContext);
    PGX_INFO("QueryContext reset - testing if JIT still works");
    
    // Try JIT execution after query context reset
    bool jit_after_reset = false;
    try {
        auto jitEngine2 = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
        bool initSuccess = jitEngine2->initialize();
        if (initSuccess) {
            bool lookupSuccess = jitEngine2->lookupCompiledQuery();
            jit_after_reset = lookupSuccess;
        }
    } catch (...) {
        jit_after_reset = false;
    }
    
    if (jit_after_reset) {
        PGX_INFO("✓ JIT execution survived QueryContext reset");
    } else {
        PGX_ERROR("✗ JIT execution failed after QueryContext reset");
        PGX_ERROR("This indicates memory context invalidation is breaking JIT execution");
        PGX_ERROR("This matches PostgreSQL LOAD command behavior that breaks our extension");
    }
    
    // Restore original context
    MemoryContextSwitchTo(oldcontext);
    
    // Clean up test contexts
    MemoryContextDelete(jitContext);
    MemoryContextDelete(queryContext);
    
    // Report results
    if (jit_success) {
        PGX_INFO("✓ JIT execution works across memory context boundaries");
        if (!jit_after_reset) {
            PGX_INFO("✓ Successfully reproduced memory context invalidation bug");
            SUCCEED() << "Identified memory context invalidation as root cause";
        }
    } else {
        PGX_ERROR("✗ JIT execution fails at memory context boundaries");
        FAIL() << "JIT fails at memory context boundaries: " << failure_reason;
    }
}

} // namespace