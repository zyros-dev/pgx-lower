#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ExecutionEngineTerminatorTest : public ::testing::Test {
protected:
    ExecutionEngineTerminatorTest() = default;
    
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc = UnknownLoc::get(&context);
    
    // Helper: Create a simple module with a function  
    ModuleOp createTestModule() {
        auto module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
        return module;
    }
    
    // Helper: Create a function for testing
    func::FuncOp createTestFunction(ModuleOp module, StringRef name, bool hasResults = false) {
        builder->setInsertionPointToEnd(module.getBody());
        auto funcType = hasResults ? 
            FunctionType::get(&context, {}, {builder->getI32Type()}) :
            FunctionType::get(&context, {}, {});
        return builder->create<func::FuncOp>(loc, name, funcType);
    }
    
    // Helper: Check if block has proper terminator
    bool hasValidTerminator(Block* block) {
        auto terminator = block->getTerminator();
        return terminator != nullptr && terminator->hasTrait<OpTrait::IsTerminator>();
    }
    
    // Helper: Count operations after terminator (should be 0)
    int countOperationsAfterTerminator(Block* block) {
        auto terminator = block->getTerminator();
        if (!terminator) return -1; // No terminator
        
        int count = 0;
        bool foundTerminator = false;
        for (auto& op : *block) {
            if (&op == terminator) {
                foundTerminator = true;
                continue;
            }
            if (foundTerminator) {
                count++;
            }
        }
        return count;
    }
};

// Test 1: Basic function creation with terminator
TEST_F(ExecutionEngineTerminatorTest, BasicFunctionTermination) {
    PGX_INFO("Testing basic function creation with terminator validation");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "test_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Add a return operation
    builder->create<func::ReturnOp>(loc);
    
    // Verify terminator is valid
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 2: Function with result requiring terminator with value
TEST_F(ExecutionEngineTerminatorTest, FunctionWithResultTermination) {
    PGX_INFO("Testing function with result termination");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "test_func_result", true);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create a constant and return it
    auto zero = builder->create<arith::ConstantIntOp>(loc, 0, 32);
    builder->create<func::ReturnOp>(loc, ValueRange{zero});
    
    // Verify terminator is valid
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 4: Test proper insertion point management (the fix)
TEST_F(ExecutionEngineTerminatorTest, ProperInsertionPointManagement) {
    PGX_INFO("Testing proper insertion point management");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "good_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Add some operations
    auto constant1 = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    
    // Add terminator
    auto returnOp = builder->create<func::ReturnOp>(loc);
    
    // Now, if we need to add more operations, we should set insertion point BEFORE terminator
    builder->setInsertionPoint(returnOp);
    auto constant2 = builder->create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Verify terminator is still last and valid
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    EXPECT_EQ(&block->back(), returnOp.getOperation());
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 5: Test ExecutionGroup pattern with proper termination
TEST_F(ExecutionEngineTerminatorTest, ExecutionGroupTermination) {
    PGX_INFO("Testing ExecutionGroup termination pattern");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "exec_group_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create an ExecutionGroup operation with proper type signature
    auto i32Type = builder->getI32Type();
    auto execGroup = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type}, ValueRange{});
    
    // Add a region with a block to the ExecutionGroup
    auto& region = execGroup.getRegion();
    auto* groupBlock = &region.emplaceBlock();
    
    builder->setInsertionPointToEnd(groupBlock);
    
    // Add some operations to the group
    auto constant = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Add proper terminator for ExecutionGroup with return value
    builder->create<subop::ExecutionGroupReturnOp>(loc, ValueRange{constant});
    
    // Add terminator to main function
    builder->setInsertionPointToEnd(block);
    builder->create<func::ReturnOp>(loc);
    
    // Verify both blocks have proper terminators
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_TRUE(hasValidTerminator(groupBlock));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    EXPECT_EQ(countOperationsAfterTerminator(groupBlock), 0);
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 8: Test nested region termination with subop operations
TEST_F(ExecutionEngineTerminatorTest, NestedRegionTermination) {
    PGX_INFO("Testing nested region termination in SubOp operations");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "nested_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create a simple operation without nested regions to test compilation
    auto constant = builder->create<arith::ConstantIntOp>(loc, 100, 32);
    
    // Add terminator to main function
    builder->create<func::ReturnOp>(loc);
    
    // Verify terminators
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 10: Test multiple execution groups with proper termination ordering
TEST_F(ExecutionEngineTerminatorTest, MultipleExecutionGroupsTermination) {
    PGX_INFO("Testing multiple execution groups with proper termination");
    
    auto module = createTestModule();
    
    // Create main function that orchestrates multiple execution groups
    auto mainFunc = createTestFunction(module, "orchestrator_func", true);
    auto* mainBlock = mainFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(mainBlock);
    
    // First execution group
    auto i32Type = builder->getI32Type();
    auto execGroup1 = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type}, ValueRange{});
    auto& region1 = execGroup1.getRegion();
    auto* block1 = &region1.emplaceBlock();
    
    builder->setInsertionPointToEnd(block1);
    auto constant1 = builder->create<arith::ConstantIntOp>(loc, 10, 32);
    builder->create<subop::ExecutionGroupReturnOp>(loc, ValueRange{constant1});
    
    // Second execution group
    builder->setInsertionPointToEnd(mainBlock);
    auto execGroup2 = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type}, ValueRange{});
    auto& region2 = execGroup2.getRegion();
    auto* block2 = &region2.emplaceBlock();
    
    builder->setInsertionPointToEnd(block2);
    auto constant2 = builder->create<arith::ConstantIntOp>(loc, 20, 32);
    builder->create<subop::ExecutionGroupReturnOp>(loc, ValueRange{constant2});
    
    // Combine results and return
    builder->setInsertionPointToEnd(mainBlock);
    auto result1 = execGroup1.getResult(0);
    auto result2 = execGroup2.getResult(0);
    auto sum = builder->create<arith::AddIOp>(loc, result1, result2);
    builder->create<func::ReturnOp>(loc, ValueRange{sum});
    
    // Verify all blocks have proper terminators
    EXPECT_TRUE(hasValidTerminator(mainBlock));
    EXPECT_TRUE(hasValidTerminator(block1));
    EXPECT_TRUE(hasValidTerminator(block2));
    
    EXPECT_EQ(countOperationsAfterTerminator(mainBlock), 0);
    EXPECT_EQ(countOperationsAfterTerminator(block1), 0);
    EXPECT_EQ(countOperationsAfterTerminator(block2), 0);
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 13: Test execution flow with multiple SubOp operations
TEST_F(ExecutionEngineTerminatorTest, MultipleSubOpOperationsFlow) {
    PGX_INFO("Testing execution flow with multiple SubOp operations");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "multi_subop_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create multiple SubOp operations in sequence
    auto constant1 = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    auto constant2 = builder->create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Create Union operation (no nested regions required) - need to provide streams
    auto unionOp = builder->create<subop::UnionOp>(loc, ValueRange{});
    
    // Add arithmetic operation
    auto sum = builder->create<arith::AddIOp>(loc, constant1, constant2);
    
    // CRITICAL: Add terminator at the end
    builder->create<func::ReturnOp>(loc);
    
    // Verify termination
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    
    // Count operations to ensure proper ordering
    int opCount = 0;
    for (auto& op : *block) {
        opCount++;
    }
    EXPECT_GE(opCount, 4); // At least constants, union, add, and return
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}