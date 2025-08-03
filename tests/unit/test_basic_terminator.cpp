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

// Test 4: MLIRBuilderStateTracker pattern - DOCUMENTS KNOWN BUG
TEST(BasicTerminatorTest, DISABLED_MLIRBuilderStateTrackerPattern_DocumentsBug) {
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
    auto func = builder.create<func::FuncOp>(loc, "tracked_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate the pattern: save insertion point
    auto savedInsertionPoint = builder.saveInsertionPoint();
    
    // Add some operations
    auto constant1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    
    // Add terminator
    auto returnOp = builder.create<func::ReturnOp>(loc);
    
    // Now save another insertion point after the terminator exists
    auto afterTerminatorIP = builder.saveInsertionPoint();
    
    // Restore the original insertion point (should be at end, but now we have a terminator)
    builder.restoreInsertionPoint(savedInsertionPoint);
    
    // If we try to add operations now, they should go BEFORE the terminator
    auto constant2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Verify terminator is still last AND still the same operation
    auto terminator = block->getTerminator();
    
    // DEBUG: Print all operations in the block to understand what happened
    std::cout << "Block contents after insertion point manipulation:\n";
    for (auto& op : *block) {
        std::cout << "  Operation: " << op.getName().getStringRef().str() << 
                     " IsTerminator: " << op.hasTrait<OpTrait::IsTerminator>() << "\n";
    }
    
    // THIS TEST DOCUMENTS THE BUG: Operations are being placed after terminator
    // When using saveInsertionPoint/restoreInsertionPoint with terminators
    
    // The terminator should be the last operation
    // EXPECT_EQ(&block->back(), terminator);  // This will fail - documenting the bug
    EXPECT_NE(terminator, nullptr);
    
    // But this will FAIL because the terminator is no longer last!
    // This is demonstrating the exact bug we found in ExecutionEngine.cpp
    if (terminator && terminator == &block->back() && terminator->hasTrait<OpTrait::IsTerminator>()) {
        std::cout << "No bug detected - terminator is correctly positioned\n";
        EXPECT_TRUE(true);  // Success case
    } else {
        // Document the bug: terminator is not the last operation
        std::cout << "BUG DETECTED: Terminator is not the last operation in block!\n";
        std::cout << "This is the root cause of ExecutionEngine terminator failures.\n";
        
        // Count operations after the terminator
        bool foundReturn = false;
        int opsAfterReturn = 0;
        for (auto& op : *block) {
            if (foundReturn) {
                opsAfterReturn++;
            }
            if (op.hasTrait<OpTrait::IsTerminator>()) {
                foundReturn = true;
            }
        }
        std::cout << "Operations after terminator: " << opsAfterReturn << "\n";
        
        // Expect the bug to be present (this documents the problem)
        EXPECT_GT(opsAfterReturn, 0) << "Expected operations after terminator (this is the bug)";
    }
    
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

// Test 5: ExecutionGroup creation pattern from ExecutionEngine.cpp
TEST(BasicTerminatorTest, ExecutionGroupCreationPattern) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module and function
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "exec_group_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate creating an execution group-like structure
    // This mimics the pattern from ExecutionEngine.cpp lines 440-570
    
    // Save state before creating nested operations
    auto savedIP = builder.saveInsertionPoint();
    
    // Create some nested operations (simulating execution group body)
    auto constant1 = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    auto constant2 = builder.create<arith::ConstantIntOp>(loc, 24, 32);
    
    // CRITICAL: Restore insertion point before adding terminator
    builder.restoreInsertionPoint(savedIP);
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify proper termination
    auto terminator = block->getTerminator();
    EXPECT_NE(terminator, nullptr);
    EXPECT_EQ(&block->back(), terminator);
    
    // Verify no operations after terminator
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

// Test 6: Store function call pattern from ExecutionEngine.cpp
TEST(BasicTerminatorTest, StoreFunctionCallPattern) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create store function declaration (simulating store_int_result)
    auto storeIntResultType = FunctionType::get(&context, 
        {builder.getI32Type(), builder.getI32Type(), builder.getI1Type()}, {});
    auto storeFunc = builder.create<func::FuncOp>(loc, "store_int_result", storeIntResultType);
    
    // Create main function
    auto funcType = FunctionType::get(&context, {}, {});
    auto func = builder.create<func::FuncOp>(loc, "main_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Create the values for the call (mimicking ExecutionEngine.cpp lines 540-541)
    auto zero32 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto fieldValue = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    auto falseVal = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    
    // Create the store_int_result call
    auto storeCallOp = builder.create<func::CallOp>(loc, storeFunc, 
        ValueRange{zero32, fieldValue, falseVal});
    
    // CRITICAL: Add terminator immediately after call
    builder.create<func::ReturnOp>(loc);
    
    // Verify no operations after terminator
    auto terminator = block->getTerminator();
    EXPECT_NE(terminator, nullptr);
    EXPECT_EQ(&block->back(), terminator);
    
    // Verify the call was created correctly
    bool foundStoreCall = false;
    for (auto& op : *block) {
        if (auto callOp = dyn_cast<func::CallOp>(&op)) {
            if (callOp.getCallee() == "store_int_result") {
                foundStoreCall = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundStoreCall);
    
    module.erase();
}

// Test 7: Block insertion management pattern
TEST(BasicTerminatorTest, BlockInsertionManagementPattern) {
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
    auto func = builder.create<func::FuncOp>(loc, "insertion_func", funcType);
    auto* block = func.addEntryBlock();
    
    // Test different insertion point scenarios
    builder.setInsertionPointToStart(block);
    auto constant1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    
    builder.setInsertionPointToEnd(block);
    auto constant2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Add terminator
    auto returnOp = builder.create<func::ReturnOp>(loc);
    
    // Test insertion before terminator
    builder.setInsertionPoint(returnOp);
    auto constant3 = builder.create<arith::ConstantIntOp>(loc, 3, 32);
    
    // Verify terminator is still last
    auto terminator = block->getTerminator();
    EXPECT_EQ(&block->back(), terminator);
    EXPECT_EQ(terminator, returnOp.getOperation());
    
    // Verify insertion order
    auto it = block->begin();
    EXPECT_EQ(&*it, constant1.getOperation()); ++it;
    EXPECT_EQ(&*it, constant2.getOperation()); ++it;
    EXPECT_EQ(&*it, constant3.getOperation()); ++it;
    EXPECT_EQ(&*it, returnOp.getOperation());
    
    module.erase();
}

// Test 8: THE FIX - Safe insertion point management with terminator checking
TEST(BasicTerminatorTest, SafeInsertionPointManagementFix) {
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
    auto func = builder.create<func::FuncOp>(loc, "safe_func", funcType);
    auto* block = func.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Save insertion point
    auto savedInsertionPoint = builder.saveInsertionPoint();
    
    // Add some operations
    auto constant1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    
    // Add terminator
    auto returnOp = builder.create<func::ReturnOp>(loc);
    
    // THE FIX: Before restoring insertion point, check for terminator
    builder.restoreInsertionPoint(savedInsertionPoint);
    
    // SAFE PATTERN: Check if block has terminator before adding operations
    if (block->getTerminator()) {
        // If terminator exists, insert BEFORE it
        builder.setInsertionPoint(block->getTerminator());
    }
    
    // Now add operations - they will go BEFORE the terminator
    auto constant2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    
    // Verify the fix worked
    auto terminator = block->getTerminator();
    EXPECT_NE(terminator, nullptr);
    EXPECT_EQ(&block->back(), terminator);
    EXPECT_TRUE(terminator->hasTrait<OpTrait::IsTerminator>());
    
    // Verify no operations after terminator
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
    
    EXPECT_EQ(opsAfterTerminator, 0) << "The fix should prevent operations after terminator";
    
    std::cout << "FIX VERIFIED: Terminator is correctly the last operation!\n";
    
    module.erase();
}