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
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ExecutionEngineTerminatorTest : public ::testing::Test {
protected:
    ExecutionEngineTerminatorTest() = default;
    
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
    
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
    auto execGroup = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type});
    
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