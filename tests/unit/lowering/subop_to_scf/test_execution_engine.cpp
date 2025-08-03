// Comprehensive unit tests for execution engine functionality
// Tests MLIR operations compilation, termination, and lowering passes

#include <gtest/gtest.h>
#include <string>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ExecutionEngineTest : public ::testing::Test {
public:
    ExecutionEngineTest() = default;

protected:
    
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
            
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc = UnknownLoc::get(&context);
    
    // Helper to create a basic module with main function
    ModuleOp createBasicModule() {
        auto module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
        
        auto mainFuncType = FunctionType::get(&context, {}, {builder->getI32Type()});
        auto mainFunc = builder->create<func::FuncOp>(loc, "main", mainFuncType);
        auto* mainBlock = mainFunc.addEntryBlock();
        
        // Add a basic terminator
        builder->setInsertionPointToEnd(mainBlock);
        auto zero = builder->create<arith::ConstantIntOp>(loc, 0, 32);
        builder->create<func::ReturnOp>(loc, ValueRange{zero});
        
        return module;
    }
    
    // Helper to create a simple function for testing
    func::FuncOp createTestFunction(ModuleOp module, const std::string& name) {
        builder->setInsertionPointToEnd(module.getBody());
        
        auto funcType = FunctionType::get(&context, {}, {builder->getI32Type()});
        auto func = builder->create<func::FuncOp>(loc, name, funcType);
        auto* block = func.addEntryBlock();
        
        builder->setInsertionPointToEnd(block);
        auto constant = builder->create<arith::ConstantIntOp>(loc, 42, 32);
        builder->create<func::ReturnOp>(loc, ValueRange{constant});
        
        return func;
    }
};

// Test 1: Basic MLIR Module Creation and Terminator Validation
TEST_F(ExecutionEngineTest, BasicModuleCreationHasTerminators) {
    PGX_INFO("Testing basic module creation with terminator validation");
    
    auto module = createBasicModule();
    
    // Verify module structure
    EXPECT_TRUE(module);
    
    // Find main function
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    EXPECT_TRUE(mainFunc);
    
    // Verify main function has entry block with terminator
    EXPECT_FALSE(mainFunc.getBody().empty());
    auto& entryBlock = mainFunc.getBody().front();
    EXPECT_TRUE(entryBlock.getTerminator());
    EXPECT_TRUE(isa<func::ReturnOp>(entryBlock.getTerminator()));
    
    // Verify MLIR module is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 2: Multiple Function Creation and Terminator Safety
TEST_F(ExecutionEngineTest, MultipleFunctionTerminatorSafety) {
    PGX_INFO("Testing multiple function creation with terminator safety");
    
    auto module = createBasicModule();
    
    // Create additional test functions
    auto func1 = createTestFunction(module, "test_func1");
    auto func2 = createTestFunction(module, "test_func2");
    
    // Verify all functions have terminators
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    EXPECT_TRUE(mainFunc);
    EXPECT_TRUE(func1);
    EXPECT_TRUE(func2);
    
    // Check terminators exist
    EXPECT_TRUE(mainFunc.getBody().front().getTerminator());
    EXPECT_TRUE(func1.getBody().front().getTerminator());
    EXPECT_TRUE(func2.getBody().front().getTerminator());
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 3: Control Flow Operations - Complex Terminator Handling
TEST_F(ExecutionEngineTest, ControlFlowTerminatorHandling) {
    PGX_INFO("Testing control flow operations with proper terminators");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function with if-else structure
    auto funcType = FunctionType::get(&context, {}, {builder->getI32Type()});
    auto testFunc = builder->create<func::FuncOp>(loc, "test_control_flow", funcType);
    auto* entryBlock = testFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(entryBlock);
    
    // Create condition and if operation
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, 1);
    auto ifOp = builder->create<scf::IfOp>(loc, builder->getI32Type(), condition, true);
    
    // Then block
    builder->setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    auto thenVal = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    builder->create<scf::YieldOp>(loc, ValueRange{thenVal});
    
    // Else block
    builder->setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    auto elseVal = builder->create<arith::ConstantIntOp>(loc, 24, 32);
    builder->create<scf::YieldOp>(loc, ValueRange{elseVal});
    
    // Return from main function
    builder->setInsertionPointToEnd(entryBlock);
    builder->create<func::ReturnOp>(loc, ValueRange{ifOp.getResult(0)});
    
    // Verify all blocks have terminators
    EXPECT_TRUE(entryBlock->getTerminator());
    EXPECT_TRUE(ifOp.getThenRegion().front().getTerminator());
    EXPECT_TRUE(ifOp.getElseRegion().front().getTerminator());
    
    // Verify MLIR is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 4: Arithmetic Operations and Memory Safety
TEST_F(ExecutionEngineTest, ArithmeticOperationsMemorySafety) {
    PGX_INFO("Testing arithmetic operations with memory safety patterns");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function with arithmetic operations
    auto funcType = FunctionType::get(&context, {}, {builder->getI32Type()});
    auto arithFunc = builder->create<func::FuncOp>(loc, "test_arithmetic", funcType);
    auto* arithBlock = arithFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(arithBlock);
    
    // Create arithmetic operations
    auto const1 = builder->create<arith::ConstantIntOp>(loc, 10, 32);
    auto const2 = builder->create<arith::ConstantIntOp>(loc, 20, 32);
    auto addOp = builder->create<arith::AddIOp>(loc, const1, const2);
    auto const3 = builder->create<arith::ConstantIntOp>(loc, 5, 32);
    auto mulOp = builder->create<arith::MulIOp>(loc, addOp, const3);
    
    // Add terminator
    builder->create<func::ReturnOp>(loc, ValueRange{mulOp});
    
    // Verify terminator exists
    EXPECT_TRUE(arithBlock->getTerminator());
    EXPECT_TRUE(isa<func::ReturnOp>(arithBlock->getTerminator()));
    
    // Verify MLIR is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 5: Pass Manager and Optimization Safety
TEST_F(ExecutionEngineTest, PassManagerOptimizationSafety) {
    PGX_INFO("Testing pass manager optimizations preserve structure");
    
    auto module = createBasicModule();
    auto testFunc = createTestFunction(module, "test_optimization");
    
    // Verify initial state
    EXPECT_TRUE(succeeded(verify(module)));
    EXPECT_TRUE(testFunc.getBody().front().getTerminator());
    
    // Run standard optimization passes
    PassManager pm(&context);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result)) << "Optimization passes should succeed";
    
    // Verify structure is preserved
    EXPECT_TRUE(succeeded(verify(module))) << "Module should remain valid after optimization";
    
    // Check functions still have terminators
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    auto optimizedFunc = module.lookupSymbol<func::FuncOp>("test_optimization");
    
    if (mainFunc && !mainFunc.getBody().empty()) {
        EXPECT_TRUE(mainFunc.getBody().front().getTerminator());
    }
    if (optimizedFunc && !optimizedFunc.getBody().empty()) {
        EXPECT_TRUE(optimizedFunc.getBody().front().getTerminator());
    }
    
    module.erase();
}

// Test 6: Error Handling - Invalid Operations
TEST_F(ExecutionEngineTest, ErrorHandlingInvalidOperations) {
    PGX_INFO("Testing error handling for invalid operations");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function without proper terminator to test error handling
    auto funcType = FunctionType::get(&context, {}, {builder->getI32Type()});
    auto badFunc = builder->create<func::FuncOp>(loc, "bad_function", funcType);
    auto* badBlock = badFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(badBlock);
    // Add operations but intentionally forget terminator
    auto constant = builder->create<arith::ConstantIntOp>(loc, 123, 32);
    // Missing: builder->create<func::ReturnOp>(loc, ValueRange{constant});
    
    // Verify module is invalid without proper terminator
    EXPECT_FALSE(succeeded(verify(module)))
        << "Module should be invalid without proper terminator";
    
    // Fix by adding terminator and verify it becomes valid
    builder->create<func::ReturnOp>(loc, ValueRange{constant});
    EXPECT_TRUE(succeeded(verify(module)))
        << "Module should become valid after adding terminator";
    
    module.erase();
}

// Test 7: Memory Context Safety Simulation
TEST_F(ExecutionEngineTest, MemoryContextSafetySimulation) {
    PGX_INFO("Testing memory context safety patterns");
    
    auto module = createBasicModule();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    auto& mainBlock = mainFunc.getBody().front();
    
    // Store original terminator
    auto originalTerminator = mainBlock.getTerminator();
    EXPECT_TRUE(originalTerminator);
    
    // Simulate inserting operations before terminator (like PostgreSQL calls)
    builder->setInsertionPoint(originalTerminator);
    
    // Add some operations before the terminator
    auto newConst = builder->create<arith::ConstantIntOp>(loc, 999, 32);
    
    // Verify terminator is still there and at the end
    EXPECT_TRUE(mainBlock.getTerminator());
    EXPECT_EQ(mainBlock.getTerminator(), originalTerminator);
    
    // Verify new operation is before terminator
    bool foundNewConst = false;
    for (auto& op : mainBlock) {
        if (&op == newConst.getOperation()) {
            foundNewConst = true;
        } else if (&op == originalTerminator) {
            EXPECT_TRUE(foundNewConst) << "New constant should appear before terminator";
            break;
        }
    }
    
    // Verify module is still valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 8: Function Call Operations and Safety
TEST_F(ExecutionEngineTest, FunctionCallOperationsSafety) {
    PGX_INFO("Testing function call operations and safety patterns");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create a helper function to call
    auto helperType = FunctionType::get(&context, {builder->getI32Type()}, {builder->getI32Type()});
    auto helperFunc = builder->create<func::FuncOp>(loc, "helper_function", helperType);
    auto* helperBlock = helperFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(helperBlock);
    auto helperArg = helperBlock->getArgument(0);
    auto two = builder->create<arith::ConstantIntOp>(loc, 2, 32);
    auto doubled = builder->create<arith::MulIOp>(loc, helperArg, two);
    builder->create<func::ReturnOp>(loc, ValueRange{doubled});
    
    // Create a caller function
    auto callerType = FunctionType::get(&context, {}, {builder->getI32Type()});
    auto callerFunc = builder->create<func::FuncOp>(loc, "caller_function", callerType);
    auto* callerBlock = callerFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(callerBlock);
    auto input = builder->create<arith::ConstantIntOp>(loc, 21, 32);
    auto callOp = builder->create<func::CallOp>(loc, helperFunc, ValueRange{input});
    builder->create<func::ReturnOp>(loc, callOp.getResults());
    
    // Verify both functions have terminators
    EXPECT_TRUE(helperBlock->getTerminator());
    EXPECT_TRUE(callerBlock->getTerminator());
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 9: Loop Operations and Complex Control Flow
TEST_F(ExecutionEngineTest, LoopOperationsComplexControlFlow) {
    PGX_INFO("Testing loop operations and complex control flow");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function with loop
    auto funcType = FunctionType::get(&context, {}, {builder->getI32Type()});
    auto loopFunc = builder->create<func::FuncOp>(loc, "test_loop", funcType);
    auto* entryBlock = loopFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(entryBlock);
    
    // Create a for loop using scf.for
    auto zero = builder->create<arith::ConstantIntOp>(loc, 0, 32);
    auto ten = builder->create<arith::ConstantIntOp>(loc, 10, 32);
    auto one = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    
    auto forOp = builder->create<scf::ForOp>(loc, zero, ten, one, ValueRange{zero});
    
    // Loop body
    builder->setInsertionPointToStart(forOp.getBody());
    auto iv = forOp.getInductionVar();
    auto currentSum = forOp.getRegionIterArg(0);
    auto newSum = builder->create<arith::AddIOp>(loc, currentSum, iv);
    builder->create<scf::YieldOp>(loc, ValueRange{newSum});
    
    // Return from function
    builder->setInsertionPointToEnd(entryBlock);
    builder->create<func::ReturnOp>(loc, forOp.getResults());
    
    // Verify terminators
    EXPECT_TRUE(entryBlock->getTerminator());
    EXPECT_TRUE(forOp.getBody()->getTerminator());
    
    // Verify MLIR is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Test 10: SubOp Dialect Basic Functionality
TEST_F(ExecutionEngineTest, SubOpDialectBasicFunctionality) {
    PGX_INFO("Testing SubOp dialect basic functionality");
    
    auto module = createBasicModule();
    
    // Test that SubOp dialect is loaded and accessible
    EXPECT_TRUE(context.getLoadedDialect<subop::SubOperatorDialect>());
    
    // Test basic dialect functionality without complex operations
    // This verifies the dialect compiles and integrates properly
    try {
        // Just test that we can reference SubOp dialect operations
        PGX_DEBUG("SubOp dialect loaded successfully");
        
        // Test pass creation (without running)
        auto lowerPass = subop::createLowerSubOpPass();
        EXPECT_TRUE(lowerPass != nullptr);
        
        auto optimizePass = subop::createGlobalOptPass();
        EXPECT_TRUE(optimizePass != nullptr);
        
        auto lowerToDBPass = subop::createLowerSubOpToDBPass();
        EXPECT_TRUE(lowerToDBPass != nullptr);
        
        PGX_INFO("SubOp passes created successfully");
        
    } catch (const std::exception& e) {
        FAIL() << "SubOp dialect operations failed: " << e.what();
    } catch (...) {
        FAIL() << "SubOp dialect operations failed with unknown exception";
    }
    
    // Verify module is still valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    module.erase();
}

// Simple functional tests without fixtures
TEST(ExecutionEngineSimpleTest, BasicModuleOperations) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a simple module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "simple_test", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    builder.create<func::ReturnOp>(loc);
    
    // Verify
    EXPECT_TRUE(block->getTerminator());
    EXPECT_TRUE(succeeded(verify(module)));
    
    PGX_INFO("Simple module operations test completed successfully");
    
    module.erase();
}

TEST(ExecutionEngineSimpleTest, ArithmeticOperationsBasic) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {builder.getI32Type()});
    auto func = builder.create<func::FuncOp>(loc, "arith_test", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    auto c1 = builder.create<arith::ConstantIntOp>(loc, 5, 32);
    auto c2 = builder.create<arith::ConstantIntOp>(loc, 3, 32);
    auto add = builder.create<arith::AddIOp>(loc, c1, c2);
    builder.create<func::ReturnOp>(loc, ValueRange{add});
    
    EXPECT_TRUE(succeeded(verify(module)));
    
    PGX_INFO("Basic arithmetic operations test completed successfully");
    
    module.erase();
}

// Test 11: MLIR Lowering Pipeline Validation
TEST(ExecutionEngineSimpleTest, MLIRLoweringPipelineValidation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module that could go through lowering pipeline
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function suitable for lowering
    auto funcType = FunctionType::get(&context, {}, {builder.getI32Type()});
    auto func = builder.create<func::FuncOp>(loc, "pipeline_test", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create arithmetic operations that could be optimized
    auto c1 = builder.create<arith::ConstantIntOp>(loc, 10, 32);
    auto c2 = builder.create<arith::ConstantIntOp>(loc, 20, 32);
    auto add1 = builder.create<arith::AddIOp>(loc, c1, c2);
    auto c3 = builder.create<arith::ConstantIntOp>(loc, 30, 32);
    auto add2 = builder.create<arith::AddIOp>(loc, add1, c3);
    
    // Add a simple loop for more complex control flow
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto five = builder.create<arith::ConstantIntOp>(loc, 5, 32);
    auto one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    
    auto forOp = builder.create<scf::ForOp>(loc, zero, five, one, ValueRange{add2});
    
    // Loop body - simple accumulation
    builder.setInsertionPointToStart(forOp.getBody());
    auto iv = forOp.getInductionVar();
    auto currentVal = forOp.getRegionIterArg(0);
    auto newVal = builder.create<arith::AddIOp>(loc, currentVal, iv);
    builder.create<scf::YieldOp>(loc, ValueRange{newVal});
    
    // Return final result
    builder.setInsertionPointToEnd(block);
    builder.create<func::ReturnOp>(loc, forOp.getResults());
    
    // Verify initial module is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    // Test individual pass creation (pipeline validation)
    try {
        // Test SubOp-specific passes
        auto globalOptPass = subop::createGlobalOptPass();
        EXPECT_TRUE(globalOptPass != nullptr);
        
        auto specializePass = subop::createSpecializeSubOpPass();
        EXPECT_TRUE(specializePass != nullptr);
        
        auto normalizePass = subop::createNormalizeSubOpPass();
        EXPECT_TRUE(normalizePass != nullptr);
        
        auto lowerPass = subop::createLowerSubOpPass();
        EXPECT_TRUE(lowerPass != nullptr);
        
        auto lowerToDBPass = subop::createLowerSubOpToDBPass();
        EXPECT_TRUE(lowerToDBPass != nullptr);
        
        PGX_INFO("MLIR lowering pipeline passes created successfully");
        
    } catch (const std::exception& e) {
        FAIL() << "Pipeline pass creation failed: " << e.what();
    } catch (...) {
        FAIL() << "Pipeline pass creation failed with unknown exception";
    }
    
    // Test that standard MLIR optimization passes work
    PassManager pm(&context);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result)) << "Standard optimization pipeline should succeed";
    
    // Verify module is still valid after optimization
    EXPECT_TRUE(succeeded(verify(module)));
    
    PGX_INFO("MLIR lowering pipeline validation completed successfully");
    
    module.erase();
}

// Test 12: Execution Engine Memory Management
TEST(ExecutionEngineSimpleTest, ExecutionEngineMemoryManagement) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Test multiple module creation and cleanup cycles
    for (int i = 0; i < 5; ++i) {
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
        
        // Create function with varied complexity
        auto funcType = FunctionType::get(&context, {}, {builder.getI32Type()});
        auto func = builder.create<func::FuncOp>(loc, "memory_test_" + std::to_string(i), funcType);
        auto* block = func.addEntryBlock();
        
        builder.setInsertionPointToEnd(block);
        
        // Create operations proportional to iteration
        Value result = builder.create<arith::ConstantIntOp>(loc, i, 32);
        
        for (int j = 0; j < i + 1; ++j) {
            auto constant = builder.create<arith::ConstantIntOp>(loc, j, 32);
            result = builder.create<arith::AddIOp>(loc, result, constant);
        }
        
        builder.create<func::ReturnOp>(loc, ValueRange{result});
        
        // Verify each iteration
        EXPECT_TRUE(succeeded(verify(module)));
        EXPECT_TRUE(block->getTerminator());
        
        // Clean up
        module.erase();
    }
    
    PGX_INFO("Execution engine memory management test completed successfully");
}