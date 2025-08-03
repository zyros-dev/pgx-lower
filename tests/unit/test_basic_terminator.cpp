#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

// Simple test to validate basic terminator patterns
TEST(BasicTerminatorTest, FunctionReturnTermination) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Add a return operation
    builder.create<func::ReturnOp>(loc);
    
    // Verify terminator exists
    auto terminator = block->getTerminator();
    EXPECT_NE(terminator, nullptr);
    EXPECT_TRUE(terminator->hasTrait<OpTrait::IsTerminator>());
    
    // Verify terminator is the last operation
    EXPECT_EQ(&block->back(), terminator);
    
    module.erase();
}

// Test what happens when we add operations after terminator (the bug)
TEST(BasicTerminatorTest, DetectOperationsAfterTerminator) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "bad_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Add a return operation
    auto returnOp = builder.create<func::ReturnOp>(loc);
    
    // Verify this is valid so far
    auto terminator = block->getTerminator();
    EXPECT_EQ(terminator, returnOp.getOperation());
    EXPECT_EQ(&block->back(), terminator);
    
    // Now simulate the bug: add operations after terminator
    // setInsertionPointToEnd will put us after the terminator
    builder.setInsertionPointToEnd(block);
    
    // This creates the bug: adding operations after terminator
    auto badConstant = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Now the terminator is NOT the last operation (this is the bug)
    EXPECT_NE(&block->back(), terminator);
    EXPECT_EQ(&block->back(), badConstant.getOperation());
    
    // Count operations after terminator
    int opsAfterTerminator = 0;
    bool foundTerminator = false;
    for (auto& op : *block) {
        if (&op == terminator) {
            foundTerminator = true;
            continue;
        }
        if (foundTerminator) {
            opsAfterTerminator++;
        }
    }
    
    // This should be 1 (the bug pattern)
    EXPECT_EQ(opsAfterTerminator, 1);
    
    module.erase();
}

// Test proper insertion point management (the fix)
TEST(BasicTerminatorTest, ProperInsertionPointManagement) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "good_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Add some operations
    auto constant1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    
    // Add terminator
    auto returnOp = builder.create<func::ReturnOp>(loc);
    
    // The FIX: If we need to add more operations, set insertion point BEFORE terminator
    builder.setInsertionPoint(returnOp);
    auto constant2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Verify terminator is still last
    auto terminator = block->getTerminator();
    EXPECT_EQ(&block->back(), terminator);
    EXPECT_EQ(terminator, returnOp.getOperation());
    
    // Count operations after terminator (should be 0)
    int opsAfterTerminator = 0;
    bool foundTerminator = false;
    for (auto& op : *block) {
        if (&op == terminator) {
            foundTerminator = true;
            continue;
        }
        if (foundTerminator) {
            opsAfterTerminator++;
        }
    }
    
    EXPECT_EQ(opsAfterTerminator, 0);
    
    module.erase();
}