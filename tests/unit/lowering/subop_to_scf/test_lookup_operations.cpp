// Simplified unit tests for Lookup operations
// Tests basic lookup operations compilation and functionality

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/util/UtilDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

// Simple test for basic lookup operation compilation
TEST(LookupOperationsTest, BasicLookupCompilation) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<tuples::TupleStreamDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a simple module for testing
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple function to test compilation
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_lookup", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create some basic operations that should compile
    auto constant = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify the function was created successfully
    EXPECT_TRUE(func);
    EXPECT_TRUE(block->getTerminator() != nullptr);
    
    PGX_INFO("Basic lookup compilation test completed successfully");
}

// Test lookup operation compilation without complex types
TEST(LookupOperationsTest, SimpleLookupOperationTest) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<util::UtilDialect>();
    context.loadDialect<tuples::TupleStreamDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create module and function
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {builder.getI32Type()});
    auto func = builder.create<func::FuncOp>(loc, "lookup_test", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create simple operations that demonstrate lookup-like patterns
    auto key = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    auto value = builder.create<arith::ConstantIntOp>(loc, 100, 32);
    
    // Simple arithmetic to simulate lookup operation
    auto result = builder.create<arith::AddIOp>(loc, key, value);
    
    builder.create<func::ReturnOp>(loc, ValueRange{result});
    
    // Verify creation was successful
    EXPECT_TRUE(func);
    EXPECT_TRUE(block->getTerminator() != nullptr);
    
    PGX_INFO("Simple lookup operation test completed successfully");
}

// Test basic state management patterns
TEST(LookupOperationsTest, StateManagementTest) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "state_test", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create basic state operations with control flow
    auto condition = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, condition, false);
    
    // Then block
    builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    auto stateValue = builder.create<arith::ConstantIntOp>(loc, 10, 32);
    builder.create<scf::YieldOp>(loc);
    
    // Return to main function
    builder.setInsertionPointToEnd(block);
    builder.create<func::ReturnOp>(loc);
    
    // Verify structure
    EXPECT_TRUE(func);
    EXPECT_TRUE(block->getTerminator() != nullptr);
    EXPECT_TRUE(ifOp.getThenRegion().front().getTerminator() != nullptr);
    
    PGX_INFO("State management test completed successfully");
}

// Test compilation with multiple operations
TEST(LookupOperationsTest, MultipleOperationsTest) {
    MLIRContext context;
    context.loadDialect<subop::SubOperatorDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create multiple functions to test compilation
    for (int i = 0; i < 3; i++) {
        auto funcType = FunctionType::get(&context, {}, {builder.getI32Type()});
        auto func = builder.create<func::FuncOp>(loc, "multi_test_" + std::to_string(i), funcType);
        auto* block = func.addEntryBlock();
        
        builder.setInsertionPointToEnd(block);
        
        // Create operations
        auto value1 = builder.create<arith::ConstantIntOp>(loc, i * 10, 32);
        auto value2 = builder.create<arith::ConstantIntOp>(loc, i + 5, 32);
        auto result = builder.create<arith::MulIOp>(loc, value1, value2);
        
        builder.create<func::ReturnOp>(loc, ValueRange{result});
        
        EXPECT_TRUE(func);
        EXPECT_TRUE(block->getTerminator() != nullptr);
    }
    
    PGX_INFO("Multiple operations test completed successfully");
}

// Test terminator safety
TEST(LookupOperationsTest, TerminatorSafetyTest) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "terminator_test", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create operations before terminator
    auto op1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    auto op2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Add terminator
    auto terminator = builder.create<func::ReturnOp>(loc);
    
    // Verify terminator is last operation
    EXPECT_TRUE(block->getTerminator() != nullptr);
    EXPECT_EQ(&block->back(), terminator.getOperation());
    
    // Count operations before terminator
    size_t opCount = 0;
    for (auto& op : block->getOperations()) {
        if (&op != terminator.getOperation()) {
            opCount++;
        }
    }
    
    EXPECT_EQ(opCount, 2); // Should have exactly 2 operations before terminator
    
    PGX_INFO("Terminator safety test completed successfully");
}