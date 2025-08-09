#include <gtest/gtest.h>
#include "execution/jit_execution_engine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

class JITExecutionEngineTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    
    void SetUp() override {
        // Load required dialects
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
    }
};

TEST_F(JITExecutionEngineTest, EngineInitialization) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Initially not initialized
    EXPECT_FALSE(engine.isInitialized());
}

TEST_F(JITExecutionEngineTest, SetOptimizationLevel) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Test setting different optimization levels
    engine.setOptimizationLevel(llvm::CodeGenOptLevel::None);
    engine.setOptimizationLevel(llvm::CodeGenOptLevel::Default);
    engine.setOptimizationLevel(llvm::CodeGenOptLevel::Aggressive);
    
    // No crash means success for now
    EXPECT_FALSE(engine.isInitialized());
}

TEST_F(JITExecutionEngineTest, SetupJITOptimizationPipeline) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Setup should succeed even before initialization
    EXPECT_TRUE(engine.setupJITOptimizationPipeline());
}

TEST_F(JITExecutionEngineTest, CompileEmptyModule) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Create an empty module
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Compilation preparation should succeed
    EXPECT_TRUE(engine.compileToLLVMIR(module));
    
    // Cleanup
    module.erase();
}

TEST_F(JITExecutionEngineTest, CompileModuleWithFunction) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Create a module with a simple function
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Create a function that returns a constant
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    // Return constant 42
    auto constant = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), constant.getResult());
    
    // Compilation preparation should succeed
    EXPECT_TRUE(engine.compileToLLVMIR(module));
    
    // Cleanup
    module.erase();
}

TEST_F(JITExecutionEngineTest, InitializeWithEmptyModule) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Create an empty module
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Initialize should succeed
    EXPECT_TRUE(engine.initialize(module));
    EXPECT_TRUE(engine.isInitialized());
    
    // Cleanup
    module.erase();
}

TEST_F(JITExecutionEngineTest, InitializeWithSimpleFunction) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Create a module with LLVM dialect operations
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Create an LLVM function that returns a constant
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(i32Type, {});
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "main", llvmFuncType);
    
    auto* block = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(block);
    
    // Return constant 42
    auto constant = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(42));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), constant.getResult());
    
    // Initialize should succeed
    EXPECT_TRUE(engine.initialize(module));
    EXPECT_TRUE(engine.isInitialized());
    
    // Cleanup
    module.erase();
}

TEST_F(JITExecutionEngineTest, DoubleInitialization) {
    pgx_lower::execution::PostgreSQLJITExecutionEngine engine;
    
    // Create a module
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // First initialization should succeed
    EXPECT_TRUE(engine.initialize(module));
    EXPECT_TRUE(engine.isInitialized());
    
    // Second initialization should also succeed (returns early)
    EXPECT_TRUE(engine.initialize(module));
    EXPECT_TRUE(engine.isInitialized());
    
    // Cleanup
    module.erase();
}