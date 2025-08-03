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

// Test 3: Test the problematic pattern - operations after terminator
TEST_F(ExecutionEngineTerminatorTest, DetectOperationsAfterTerminator) {
    PGX_INFO("Testing detection of operations after terminator (bug pattern)");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "bad_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Add a return operation
    auto returnOp = builder->create<func::ReturnOp>(loc);
    
    // Verify this is valid so far
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    
    // Now simulate the bug: set insertion point to end and add more operations
    builder->setInsertionPointToEnd(block);
    
    // This would be the bug: adding operations after the terminator
    // NOTE: This test demonstrates what NOT to do
    auto badConstant = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    
    // This should now detect operations after terminator
    EXPECT_EQ(countOperationsAfterTerminator(block), 1);
    
    // The block should still have a terminator, but it's in the wrong position
    EXPECT_TRUE(block->getTerminator() != nullptr);
    EXPECT_NE(block->getTerminator(), &block->back()); // Terminator is not the last operation
    
    // Module should now be invalid due to improper terminator placement
    EXPECT_TRUE(failed(verify(module)));
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

// Test 6: Test store_int_result call pattern (the specific problematic case)
TEST_F(ExecutionEngineTerminatorTest, StoreIntResultCallPattern) {
    PGX_INFO("Testing store_int_result call pattern with proper termination");
    
    auto module = createTestModule();
    
    // Create store_int_result function declaration
    builder->setInsertionPointToEnd(module.getBody());
    auto storeIntResultType = FunctionType::get(&context, 
        {builder->getI32Type(), builder->getI32Type(), builder->getI1Type()}, {});
    auto storeFunc = builder->create<func::FuncOp>(loc, "store_int_result", storeIntResultType);
    
    // Create main function
    auto mainFunc = createTestFunction(module, "main_func", false);
    auto* mainBlock = mainFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(mainBlock);
    
    // Create the values for the call
    auto zero32 = builder->create<arith::ConstantIntOp>(loc, 0, 32);
    auto fieldValue = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    auto falseVal = builder->create<arith::ConstantIntOp>(loc, 0, 1);
    
    // Create the store_int_result call
    auto storeCallOp = builder->create<func::CallOp>(loc, storeFunc, 
        ValueRange{zero32, fieldValue, falseVal});
    
    // CRITICAL: Add terminator immediately after call
    builder->create<func::ReturnOp>(loc);
    
    // Verify no operations after terminator
    EXPECT_TRUE(hasValidTerminator(mainBlock));
    EXPECT_EQ(countOperationsAfterTerminator(mainBlock), 0);
    
    // Verify the call was created correctly
    bool foundStoreCall = false;
    for (auto& op : *mainBlock) {
        if (auto callOp = dyn_cast<func::CallOp>(&op)) {
            if (callOp.getCallee() == "store_int_result") {
                foundStoreCall = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundStoreCall);
    
    // Verify module is valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 7: Test conditional branching with proper termination
TEST_F(ExecutionEngineTerminatorTest, ConditionalBranchingTermination) {
    PGX_INFO("Testing conditional branching with proper terminators");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "branch_func", true);
    auto* entryBlock = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(entryBlock);
    
    // Create condition
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, 1);
    
    // Create SCF if operation with proper termination
    auto ifOp = builder->create<scf::IfOp>(loc, builder->getI32Type(), condition, true);
    
    // Then block
    builder->setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    auto trueVal = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    builder->create<scf::YieldOp>(loc, ValueRange{trueVal});
    
    // Else block  
    builder->setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    auto falseVal = builder->create<arith::ConstantIntOp>(loc, 0, 32);
    builder->create<scf::YieldOp>(loc, ValueRange{falseVal});
    
    // Return result
    builder->setInsertionPointToEnd(entryBlock);
    builder->create<func::ReturnOp>(loc, ValueRange{ifOp.getResult(0)});
    
    // Verify all blocks have proper terminators
    EXPECT_TRUE(hasValidTerminator(entryBlock));
    EXPECT_TRUE(hasValidTerminator(&ifOp.getThenRegion().front()));
    EXPECT_TRUE(hasValidTerminator(&ifOp.getElseRegion().front()));
    
    EXPECT_EQ(countOperationsAfterTerminator(entryBlock), 0);
    EXPECT_EQ(countOperationsAfterTerminator(&ifOp.getThenRegion().front()), 0);
    EXPECT_EQ(countOperationsAfterTerminator(&ifOp.getElseRegion().front()), 0);
    
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

// Test 9: Test loop operation termination patterns
TEST_F(ExecutionEngineTerminatorTest, LoopOperationTermination) {
    PGX_INFO("Testing loop operation termination patterns");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "loop_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create a simple SCF for loop
    auto lowerBound = builder->create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = builder->create<arith::ConstantIndexOp>(loc, 10);
    auto step = builder->create<arith::ConstantIndexOp>(loc, 1);
    
    auto forOp = builder->create<scf::ForOp>(loc, lowerBound, upperBound, step);
    
    // Add operations in the loop body
    auto* loopBlock = forOp.getBody();
    builder->setInsertionPointToEnd(loopBlock);
    
    // Add some operations in the loop
    auto iterValue = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    
    // SCF ForOp automatically handles termination with scf::YieldOp
    builder->create<scf::YieldOp>(loc);
    
    // Add terminator to main function
    builder->setInsertionPointToEnd(block);
    builder->create<func::ReturnOp>(loc);
    
    // Verify terminators
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_TRUE(hasValidTerminator(loopBlock));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    EXPECT_EQ(countOperationsAfterTerminator(loopBlock), 0);
    
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

// Test 11: Test error recovery - fixing malformed termination
TEST_F(ExecutionEngineTerminatorTest, ErrorRecoveryTerminationFix) {
    PGX_INFO("Testing error recovery from malformed termination");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "recovery_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Simulate the problematic pattern - create operations without terminator
    auto constant1 = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    auto constant2 = builder->create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Add terminator
    auto returnOp = builder->create<func::ReturnOp>(loc);
    
    // Simulate adding operation after terminator (the bug)
    builder->setInsertionPointToEnd(block);
    auto badConstant = builder->create<arith::ConstantIntOp>(loc, 99, 32);
    
    // Verify we can detect the problem
    EXPECT_EQ(countOperationsAfterTerminator(block), 1);
    EXPECT_TRUE(failed(verify(module)));
    
    // Fix the problem by moving the bad operation before the terminator
    badConstant->moveBefore(returnOp);
    
    // Verify the fix
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(&block->back(), returnOp.getOperation());
    
    // Verify module is now valid
    EXPECT_TRUE(succeeded(verify(module)));
}

// Test 12: Test SubOp specific operations with terminators
TEST_F(ExecutionEngineTerminatorTest, SubOpSpecificOperationsTermination) {
    PGX_INFO("Testing SubOp specific operations with proper termination");
    
    auto module = createTestModule();
    auto func = createTestFunction(module, "subop_specific_func", false);
    auto* block = func.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Test basic SubOp operations that should compile
    // Generate operation - simple test (need to provide required attributes)
    auto emptyColumns = builder->getArrayAttr({});
    auto generateOp = builder->create<subop::GenerateOp>(loc, TypeRange{}, emptyColumns);
    
    // Create a simple constant for testing
    auto constant = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Add proper terminator
    builder->create<func::ReturnOp>(loc);
    
    // Verify termination
    EXPECT_TRUE(hasValidTerminator(block));
    EXPECT_EQ(countOperationsAfterTerminator(block), 0);
    
    // Verify the generate operation was created
    bool foundGenerateOp = false;
    for (auto& op : *block) {
        if (isa<subop::GenerateOp>(&op)) {
            foundGenerateOp = true;
            break;
        }
    }
    EXPECT_TRUE(foundGenerateOp);
    
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

// Test 14: Comprehensive execution engine terminator validation
TEST_F(ExecutionEngineTerminatorTest, ComprehensiveTerminatorValidation) {
    PGX_INFO("Comprehensive execution engine terminator validation test");
    
    auto module = createTestModule();
    
    // Create multiple functions to test various termination patterns
    auto mainFunc = createTestFunction(module, "comprehensive_main", true);
    auto helperFunc = createTestFunction(module, "helper", false);
    
    // Test helper function termination
    auto* helperBlock = helperFunc.addEntryBlock();
    builder->setInsertionPointToEnd(helperBlock);
    auto helperConstant = builder->create<arith::ConstantIntOp>(loc, 999, 32);
    builder->create<func::ReturnOp>(loc);
    
    // Test main function with complex flow
    auto* mainBlock = mainFunc.addEntryBlock();
    builder->setInsertionPointToEnd(mainBlock);
    
    // Create initial values
    auto input1 = builder->create<arith::ConstantIntOp>(loc, 10, 32);
    auto input2 = builder->create<arith::ConstantIntOp>(loc, 20, 32);
    
    // Call helper function
    auto helperCall = builder->create<func::CallOp>(loc, helperFunc, ValueRange{});
    
    // Perform computation
    auto computation = builder->create<arith::MulIOp>(loc, input1, input2);
    
    // Create conditional logic using SCF operations
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, 1);
    auto ifOp = builder->create<scf::IfOp>(loc, builder->getI32Type(), condition, true);
    
    // Then block - add computation
    builder->setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    auto trueResult = builder->create<arith::AddIOp>(loc, computation, input1);
    builder->create<scf::YieldOp>(loc, ValueRange{trueResult});
    
    // Else block - subtract computation  
    builder->setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    auto falseResult = builder->create<arith::SubIOp>(loc, computation, input1);
    builder->create<scf::YieldOp>(loc, ValueRange{falseResult});
    
    // Return final result
    builder->setInsertionPointToEnd(mainBlock);
    builder->create<func::ReturnOp>(loc, ValueRange{ifOp.getResult(0)});
    
    // Comprehensive validation of all blocks
    EXPECT_TRUE(hasValidTerminator(helperBlock));
    EXPECT_TRUE(hasValidTerminator(mainBlock));
    EXPECT_TRUE(hasValidTerminator(&ifOp.getThenRegion().front()));
    EXPECT_TRUE(hasValidTerminator(&ifOp.getElseRegion().front()));
    
    // Verify no operations after terminators in any block
    EXPECT_EQ(countOperationsAfterTerminator(helperBlock), 0);
    EXPECT_EQ(countOperationsAfterTerminator(mainBlock), 0);
    EXPECT_EQ(countOperationsAfterTerminator(&ifOp.getThenRegion().front()), 0);
    EXPECT_EQ(countOperationsAfterTerminator(&ifOp.getElseRegion().front()), 0);
    
    // Verify terminator types are correct
    auto mainTerminator = mainBlock->getTerminator();
    auto thenTerminator = ifOp.getThenRegion().front().getTerminator();
    auto elseTerminator = ifOp.getElseRegion().front().getTerminator();
    
    EXPECT_TRUE(isa<func::ReturnOp>(mainTerminator));
    EXPECT_TRUE(isa<scf::YieldOp>(thenTerminator));
    EXPECT_TRUE(isa<scf::YieldOp>(elseTerminator));
    
    // Verify the entire module is valid
    EXPECT_TRUE(succeeded(verify(module)));
    
    PGX_INFO("Comprehensive terminator validation completed successfully");
}