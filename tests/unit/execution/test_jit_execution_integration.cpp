// Phase 4g-2c: Integration tests for JIT execution with header isolation
#include <gtest/gtest.h>
#include "execution/jit_execution_interface.h"
#include "execution/jit_execution_engine.h"
#include "execution/logging.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

// Forward declare module handle functions
extern "C" {
    struct ModuleHandle* pgx_jit_create_module_handle(void* mlir_module_ptr);
    void pgx_jit_destroy_module_handle(struct ModuleHandle* handle);
}

// Mock structures for PostgreSQL types
struct MockEState {
    int dummy_value = 42;
};

struct MockDestReceiver {
    int result_count = 0;
    bool results_ready = false;
};

class JITExecutionIntegrationTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    
    void SetUp() override {
        // Load required dialects
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
    }
};

// Test the executeCompiledQuery method directly
TEST_F(JITExecutionIntegrationTest, ExecuteCompiledQuery) {
    PGX_DEBUG("Testing executeCompiledQuery with mock functions");
    
    // Create a simple LLVM module with a main function
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Create an LLVM function that returns 0 (success)
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(i32Type, {voidPtrType, voidPtrType});
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "main", llvmFuncType);
    
    auto* block = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(block);
    
    // Return constant 0 (success)
    auto constant = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(0));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), constant.getResult());
    
    // Create JIT engine and test executeCompiledQuery
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    EXPECT_TRUE(engine.initialize(module));
    
    MockEState estate;
    MockDestReceiver dest;
    
    // Execute the compiled query
    bool result = engine.executeCompiledQuery(&estate, &dest);
    EXPECT_TRUE(result) << "executeCompiledQuery should succeed with return value 0";
    
    // Cleanup
    module.erase();
}

// Test run_mlir_with_dest_receiver functionality
TEST_F(JITExecutionIntegrationTest, RunMLIRWithDestReceiver) {
    PGX_DEBUG("Testing run_mlir_with_dest_receiver through C interface");
    
    // Create a simple module
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(i32Type, {voidPtrType, voidPtrType});
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "compiled_query", llvmFuncType);
    
    auto* block = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(block);
    
    // Return constant 0 (success)
    auto constant = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(0));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), constant.getResult());
    
    // Test through C interface
    auto moduleHandle = pgx_jit_create_module_handle(&module);
    ASSERT_NE(moduleHandle, nullptr) << "Should create module handle";
    
    auto execHandle = pgx_jit_create_execution_handle(moduleHandle);
    pgx_jit_destroy_module_handle(moduleHandle);
    
    ASSERT_NE(execHandle, nullptr) << "Should create execution handle";
    
    MockEState estate;
    MockDestReceiver dest;
    
    int result = pgx_jit_execute_query(execHandle, &estate, &dest);
    EXPECT_EQ(result, 0) << "JIT execution should succeed";
    
    pgx_jit_destroy_execution_handle(execHandle);
    
    // Cleanup
    module.erase();
}

// Test complete pipeline execution simulation
TEST_F(JITExecutionIntegrationTest, CompletePipelineExecution) {
    PGX_DEBUG("Testing complete Test 1 pipeline execution");
    
    // Create a more complex module simulating Test 1
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Create main function
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(i32Type, {voidPtrType, voidPtrType});
    
    // Create compiled_query function first
    auto queryFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "compiled_query", llvmFuncType);
    
    auto* queryBlock = queryFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(queryBlock);
    
    // Return 0 from compiled_query
    auto zero = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(0));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), zero.getResult());
    
    // Create main function
    builder.setInsertionPointToEnd(module.getBody());
    auto mainFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "main", llvmFuncType);
    
    auto* mainBlock = mainFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(mainBlock);
    
    // Just return 0 from main for now - simplify the test
    auto mainZero = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(0));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mainZero.getResult());
    
    // Test execution
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    EXPECT_TRUE(engine.initialize(module));
    
    MockEState estate;
    MockDestReceiver dest;
    
    bool result = engine.executeCompiledQuery(&estate, &dest);
    EXPECT_TRUE(result) << "Complete pipeline execution should succeed";
    
    // Cleanup
    module.erase();
}

// Test error handling in JIT execution
TEST_F(JITExecutionIntegrationTest, ErrorHandling) {
    PGX_DEBUG("Testing JIT execution error handling");
    
    // Test null module handle
    auto execHandle = pgx_jit_create_execution_handle(nullptr);
    EXPECT_EQ(execHandle, nullptr) << "Should fail with null module handle";
    
    const char* error = pgx_jit_get_last_error();
    EXPECT_NE(error, nullptr) << "Should have error message";
    if (error) {
        PGX_DEBUG(std::string("Error message: ") + error);
    }
    
    // Test null parameters to execute
    MockEState estate;
    int result = pgx_jit_execute_query(nullptr, &estate, nullptr);
    EXPECT_NE(result, 0) << "Should fail with null execution handle";
    
    // Create a module that will fail to execute
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Create function with wrong signature
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(i32Type, {});  // No parameters
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "wrong_signature", llvmFuncType);
    
    auto* block = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(block);
    
    auto constant = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(1));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), constant.getResult());
    
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    EXPECT_TRUE(engine.initialize(module));
    
    // This should fail because there's no "main" or "compiled_query" function
    bool execResult = engine.executeCompiledQuery(&estate, nullptr);
    EXPECT_FALSE(execResult) << "Should fail without proper entry function";
    
    // Cleanup
    module.erase();
}

// Test memory context setup
TEST_F(JITExecutionIntegrationTest, MemoryContextSetup) {
    PGX_DEBUG("Testing memory context setup");
    
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // In unit test environment, this should succeed (no-op)
    bool result = engine.setupMemoryContexts();
    EXPECT_TRUE(result) << "Memory context setup should succeed in unit tests";
}

// Test runtime function registration
TEST_F(JITExecutionIntegrationTest, RuntimeFunctionRegistration) {
    PGX_DEBUG("Testing runtime function registration");
    
    // Create a module that uses runtime functions
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    EXPECT_TRUE(engine.initialize(module));
    
    // Register runtime functions - should not crash
    engine.registerPostgreSQLRuntimeFunctions();
    
    // The actual function calls would be tested in integration tests
    // Here we just verify registration doesn't crash
    
    // Cleanup
    module.erase();
}

// Test optimization pipeline configuration
TEST_F(JITExecutionIntegrationTest, OptimizationConfiguration) {
    PGX_DEBUG("Testing JIT optimization configuration");
    
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Test different optimization levels
    engine.setOptimizationLevel(llvm::CodeGenOptLevel::None);
    EXPECT_TRUE(engine.setupJITOptimizationPipeline());
    
    engine.setOptimizationLevel(llvm::CodeGenOptLevel::Default);
    EXPECT_TRUE(engine.setupJITOptimizationPipeline());
    
    engine.setOptimizationLevel(llvm::CodeGenOptLevel::Aggressive);
    EXPECT_TRUE(engine.setupJITOptimizationPipeline());
    
    // Create and compile a module with optimization
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(i32Type, {});
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "optimize_me", llvmFuncType);
    
    auto* block = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(block);
    
    // Create some operations that can be optimized
    auto c1 = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(5));
    auto c2 = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(10));
    auto add = builder.create<mlir::LLVM::AddOp>(
        builder.getUnknownLoc(), c1.getResult(), c2.getResult());
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), add.getResult());
    
    EXPECT_TRUE(engine.initialize(module));
    
    // Cleanup
    module.erase();
}