#include <gtest/gtest.h>
#include "pgx_lower/execution/jit_execution_engine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/TargetSelect.h"

using namespace pgx_lower::execution;

class JITExecutionEngineTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    std::unique_ptr<PostgreSQLJITExecutionEngine> engine;
    
    void SetUp() override {
        // Initialize MLIR context with necessary dialects
        context.loadDialect<mlir::LLVM::LLVMDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        
        // Create JIT execution engine
        engine = std::make_unique<PostgreSQLJITExecutionEngine>();
    }
    
    // Helper to create a simple LLVM module for testing
    mlir::ModuleOp createTestModule() {
        mlir::OpBuilder builder(&context);
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        builder.setInsertionPointToStart(module.getBody());
        
        // Create a simple main function that returns void
        auto voidType = mlir::LLVM::LLVMVoidType::get(&context);
        auto funcType = mlir::LLVM::LLVMFunctionType::get(voidType, {});
        auto mainFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
            builder.getUnknownLoc(), 
            "main", 
            funcType
        );
        
        // Add a basic block with return
        auto* entryBlock = mainFunc.addEntryBlock(builder);
        builder.setInsertionPointToStart(entryBlock);
        builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
        
        return module;
    }
};

TEST_F(JITExecutionEngineTest, Initialization) {
    ASSERT_NE(engine, nullptr);
    EXPECT_FALSE(engine->isInitialized());
}

TEST_F(JITExecutionEngineTest, InitializeWithValidModule) {
    auto module = createTestModule();
    ASSERT_TRUE(module);
    
    bool result = engine->initialize(module);
    EXPECT_TRUE(result);
    EXPECT_TRUE(engine->isInitialized());
}

TEST_F(JITExecutionEngineTest, InitializeWithNullModule) {
    mlir::ModuleOp nullModule;
    
    bool result = engine->initialize(nullModule);
    EXPECT_FALSE(result);
    EXPECT_FALSE(engine->isInitialized());
}

TEST_F(JITExecutionEngineTest, DoubleInitialization) {
    auto module = createTestModule();
    
    bool result1 = engine->initialize(module);
    EXPECT_TRUE(result1);
    
    // Second initialization should be handled gracefully
    bool result2 = engine->initialize(module);
    EXPECT_TRUE(result2); // Should return true but warn
    EXPECT_TRUE(engine->isInitialized());
}

TEST_F(JITExecutionEngineTest, SetOptimizationLevel) {
    engine->setOptimizationLevel(llvm::CodeGenOptLevel::None);
    
    auto module = createTestModule();
    bool result = engine->initialize(module);
    EXPECT_TRUE(result);
}

TEST_F(JITExecutionEngineTest, SetupJITOptimizationPipeline) {
    bool result = engine->setupJITOptimizationPipeline();
    EXPECT_TRUE(result);
}

TEST_F(JITExecutionEngineTest, CompileToLLVMIR) {
    auto module = createTestModule();
    
    bool result = engine->compileToLLVMIR(module);
    EXPECT_TRUE(result);
}

TEST_F(JITExecutionEngineTest, CompileToLLVMIRWithNullModule) {
    mlir::ModuleOp nullModule;
    
    bool result = engine->compileToLLVMIR(nullModule);
    EXPECT_FALSE(result);
}

TEST_F(JITExecutionEngineTest, SetupMemoryContexts) {
    // In unit test environment, this should succeed
    bool result = engine->setupMemoryContexts();
    EXPECT_TRUE(result);
}

TEST_F(JITExecutionEngineTest, ExecuteWithoutInitialization) {
    // Create dummy pointers for estate and dest
    int dummy_estate = 0;
    int dummy_dest = 0;
    
    bool result = engine->executeCompiledQuery(&dummy_estate, &dummy_dest);
    EXPECT_FALSE(result);
}

TEST_F(JITExecutionEngineTest, ExecuteWithNullParameters) {
    auto module = createTestModule();
    engine->initialize(module);
    
    bool result = engine->executeCompiledQuery(nullptr, nullptr);
    EXPECT_FALSE(result);
}

// Test for the WrappedExecutionEngine pattern
TEST_F(JITExecutionEngineTest, LingoDPatternExecution) {
    // Create a module with LLVM dialect operations
    auto module = createTestModule();
    ASSERT_TRUE(module);
    
    // Initialize the engine
    bool initResult = engine->initialize(module);
    EXPECT_TRUE(initResult);
    
    // Setup memory contexts
    bool memResult = engine->setupMemoryContexts();
    EXPECT_TRUE(memResult);
    
    // Try to execute (will fail without proper runtime setup, but tests the flow)
    int dummy_estate = 0;
    int dummy_dest = 0;
    
    // This tests the execution path and should succeed with our simple test module
    bool execResult = engine->executeCompiledQuery(&dummy_estate, &dummy_dest);
    // The execution should succeed with our LingoDB pattern implementation
    EXPECT_TRUE(execResult);
}

// Test optimization levels
TEST_F(JITExecutionEngineTest, OptimizationLevels) {
    std::vector<llvm::CodeGenOptLevel> levels = {
        llvm::CodeGenOptLevel::None,
        llvm::CodeGenOptLevel::Less,
        llvm::CodeGenOptLevel::Default,
        llvm::CodeGenOptLevel::Aggressive
    };
    
    for (auto level : levels) {
        auto testEngine = std::make_unique<PostgreSQLJITExecutionEngine>();
        testEngine->setOptimizationLevel(level);
        
        auto module = createTestModule();
        bool result = testEngine->initialize(module);
        EXPECT_TRUE(result) << "Failed with optimization level: " << static_cast<int>(level);
    }
}

// Test module validation
TEST_F(JITExecutionEngineTest, ModuleValidation) {
    // Create an invalid module (empty)
    mlir::OpBuilder builder(&context);
    auto emptyModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    // This should succeed even with empty module
    bool result = engine->compileToLLVMIR(emptyModule);
    EXPECT_TRUE(result); // Empty module is still valid
}

// Test runtime function registration
TEST_F(JITExecutionEngineTest, RuntimeFunctionRegistration) {
    auto module = createTestModule();
    
    // Initialize should register runtime functions
    bool result = engine->initialize(module);
    EXPECT_TRUE(result);
    
    // Call registerPostgreSQLRuntimeFunctions explicitly (should be safe to call again)
    engine->registerPostgreSQLRuntimeFunctions();
    
    // No crash means success
    EXPECT_TRUE(engine->isInitialized());
}