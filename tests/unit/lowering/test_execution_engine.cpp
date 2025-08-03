#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ExecutionEngineTest : public ::testing::Test {
protected:
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
    
    // Helper to create a basic module with main function
    ModuleOp createBasicModule() {
        auto module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
        
        auto mainFuncType = builder->getFunctionType({}, builder->getI32Type());
        auto mainFunc = builder->create<func::FuncOp>(loc, "main", mainFuncType);
        auto* mainBlock = mainFunc.addEntryBlock();
        
        // Add a basic terminator
        builder->setInsertionPointToEnd(mainBlock);
        auto zero = builder->create<arith::ConstantIntOp>(loc, 0, 32);
        builder->create<func::ReturnOp>(loc, ValueRange{zero});
        
        return module;
    }
    
    // Helper to create ExecutionGroupOp with proper structure
    subop::ExecutionGroupOp createExecutionGroup(ModuleOp module) {
        builder->setInsertionPointToEnd(module.getBody());
        
        auto i32Type = builder->getI32Type();
        auto executionGroup = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type});
        
        auto* execBlock = &executionGroup.getRegion().emplaceBlock();
        builder->setInsertionPointToEnd(execBlock);
        
        // Create a simple constant for return
        auto constant = builder->create<arith::ConstantIntOp>(loc, 42, 32);
        builder->create<subop::ExecutionGroupReturnOp>(loc, ValueRange{constant});
        
        return executionGroup;
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
}

// Test 2: ExecutionGroupOp Structure Validation
TEST_F(ExecutionEngineTest, ExecutionGroupStructureValidation) {
    PGX_INFO("Testing ExecutionGroupOp structure validation");
    
    auto module = createBasicModule();
    auto execGroup = createExecutionGroup(module);
    
    // Verify ExecutionGroupOp structure
    EXPECT_TRUE(execGroup);
    EXPECT_EQ(execGroup.getNumResults(), 1);
    
    // Verify region has block with terminator
    EXPECT_EQ(execGroup.getRegion().getBlocks().size(), 1);
    auto& execBlock = execGroup.getRegion().front();
    EXPECT_TRUE(execBlock.getTerminator());
    EXPECT_TRUE(isa<subop::ExecutionGroupReturnOp>(execBlock.getTerminator()));
}

// Test 3: SubOp Pass Execution - Terminator Preservation
TEST_F(ExecutionEngineTest, SubOpPassPreservesTerminators) {
    PGX_INFO("Testing SubOp lowering pass preserves terminators");
    
    auto module = createBasicModule();
    auto execGroup = createExecutionGroup(module);
    
    // Run the SubOp lowering pass
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result));
    
    // Verify main function still has terminator after pass
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    EXPECT_TRUE(mainFunc);
    
    if (!mainFunc.getBody().empty()) {
        auto& entryBlock = mainFunc.getBody().front();
        EXPECT_TRUE(entryBlock.getTerminator()) 
            << "Main function lost terminator after SubOp lowering pass";
    }
    
    // Verify MLIR is still valid
    EXPECT_TRUE(succeeded(verify(module))) 
        << "Module failed MLIR verification after SubOp lowering";
}

// Test 4: Complex ExecutionGroupOp with Multiple Operations
TEST_F(ExecutionEngineTest, ComplexExecutionGroupTerminatorHandling) {
    PGX_INFO("Testing complex ExecutionGroupOp with multiple operations");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create ExecutionGroupOp with GetExternalOp and ScanRefsOp
    auto i32Type = builder->getI32Type();
    auto tableType = subop::TableType::get(&context, TupleType::get(&context, {i32Type}));
    
    auto executionGroup = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type});
    auto* execBlock = &executionGroup.getRegion().emplaceBlock();
    builder->setInsertionPointToEnd(execBlock);
    
    // Add GetExternalOp
    auto getExternal = builder->create<subop::GetExternalOp>(loc, tableType, 
        builder->getStringAttr("test_table"));
    
    // Add ScanRefsOp  
    auto scanRefs = builder->create<subop::ScanRefsOp>(loc, i32Type, getExternal.getResult());
    
    // Proper terminator
    builder->create<subop::ExecutionGroupReturnOp>(loc, ValueRange{scanRefs.getResult()});
    
    // Verify structure before lowering
    EXPECT_TRUE(execBlock->getTerminator());
    EXPECT_TRUE(isa<subop::ExecutionGroupReturnOp>(execBlock->getTerminator()));
    
    // Run lowering pass
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result));
    
    // Verify module is still valid
    EXPECT_TRUE(succeeded(verify(module)))
        << "Module failed verification after complex ExecutionGroup lowering";
}

// Test 5: Function Creation and Terminator Safety
TEST_F(ExecutionEngineTest, FunctionCreationTerminatorSafety) {
    PGX_INFO("Testing function creation maintains terminator safety");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create multiple functions to test terminator management
    auto voidFuncType = builder->getFunctionType({}, {});
    auto func1 = builder->create<func::FuncOp>(loc, "test_func1", voidFuncType);
    auto* block1 = func1.addEntryBlock();
    
    auto i32FuncType = builder->getFunctionType({}, builder->getI32Type());
    auto func2 = builder->create<func::FuncOp>(loc, "test_func2", i32FuncType);
    auto* block2 = func2.addEntryBlock();
    
    // Add proper terminators
    builder->setInsertionPointToEnd(block1);
    builder->create<func::ReturnOp>(loc);
    
    builder->setInsertionPointToEnd(block2);
    auto retVal = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    builder->create<func::ReturnOp>(loc, ValueRange{retVal});
    
    // Verify both functions have terminators
    EXPECT_TRUE(block1->getTerminator());
    EXPECT_TRUE(block2->getTerminator());
    EXPECT_TRUE(isa<func::ReturnOp>(block1->getTerminator()));
    EXPECT_TRUE(isa<func::ReturnOp>(block2->getTerminator()));
    
    // Run lowering and verify terminators preserved
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result));
    
    // Check functions still exist and have terminators
    auto func1After = module.lookupSymbol<func::FuncOp>("test_func1");
    auto func2After = module.lookupSymbol<func::FuncOp>("test_func2");
    
    if (func1After && !func1After.getBody().empty()) {
        EXPECT_TRUE(func1After.getBody().front().getTerminator());
    }
    if (func2After && !func2After.getBody().empty()) {
        EXPECT_TRUE(func2After.getBody().front().getTerminator());
    }
}

// Test 6: Error Handling - Invalid Operations
TEST_F(ExecutionEngineTest, ErrorHandlingInvalidOperations) {
    PGX_INFO("Testing error handling for invalid operations");
    
    auto module = createBasicModule();
    
    // Create ExecutionGroupOp without proper terminator
    builder->setInsertionPointToEnd(module.getBody());
    auto i32Type = builder->getI32Type();
    auto badExecGroup = builder->create<subop::ExecutionGroupOp>(loc, TypeRange{i32Type});
    auto* badBlock = &badExecGroup.getRegion().emplaceBlock();
    
    builder->setInsertionPointToEnd(badBlock);
    // Intentionally don't add terminator to test error handling
    auto constant = builder->create<arith::ConstantIntOp>(loc, 123, 32);
    // Missing: builder->create<subop::ExecutionGroupReturnOp>(loc, ValueRange{constant});
    
    // Verify module is initially invalid
    EXPECT_FALSE(succeeded(verify(module)))
        << "Module should be invalid without proper terminator";
}

// Test 7: Memory Context Safety Simulation
TEST_F(ExecutionEngineTest, MemoryContextSafetySiumlation) {
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
        if (&op == newConst) {
            foundNewConst = true;
        } else if (&op == originalTerminator) {
            EXPECT_TRUE(foundNewConst) << "New constant should appear before terminator";
            break;
        }
    }
}

// Test 8: MLIR Verification After Lowering
TEST_F(ExecutionEngineTest, MLIRVerificationAfterLowering) {
    PGX_INFO("Testing MLIR verification passes after lowering");
    
    auto module = createBasicModule();
    auto execGroup = createExecutionGroup(module);
    
    // Verify before lowering
    EXPECT_TRUE(succeeded(verify(module))) 
        << "Module should be valid before lowering";
    
    // Run SubOp lowering
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result)) 
        << "SubOp lowering pass should succeed";
    
    // Verify after lowering  
    EXPECT_TRUE(succeeded(verify(module)))
        << "Module should remain valid after SubOp lowering";
    
    // Run additional verification passes
    PassManager pm2(&context);
    pm2.addPass(createCanonicalizerPass());
    pm2.addPass(createCSEPass());
    
    auto result2 = pm2.run(module);
    EXPECT_TRUE(succeeded(result2))
        << "Additional optimization passes should succeed";
    
    EXPECT_TRUE(succeeded(verify(module)))
        << "Module should remain valid after optimization passes";
}

// Test 9: Block Terminator Edge Cases
TEST_F(ExecutionEngineTest, BlockTerminatorEdgeCases) {
    PGX_INFO("Testing block terminator edge cases");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function with nested control flow
    auto funcType = builder->getFunctionType({}, builder->getI32Type());
    auto testFunc = builder->create<func::FuncOp>(loc, "test_nested", funcType);
    auto* entryBlock = testFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(entryBlock);
    
    // Create if-else structure
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    auto ifOp = builder->create<scf::IfOp>(loc, builder->getI32Type(), condition, 
        /*withElseRegion=*/true);
    
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
    EXPECT_TRUE(succeeded(verify(module)))
        << "Nested control flow should be valid";
    
    // Run lowering pass
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result));
    
    // Verify still valid after lowering
    EXPECT_TRUE(succeeded(verify(module)))
        << "Nested control flow should remain valid after lowering";
}

// Test 10: PostgreSQL Runtime Call Pattern
TEST_F(ExecutionEngineTest, PostgreSQLRuntimeCallPattern) {
    PGX_INFO("Testing PostgreSQL runtime call pattern");
    
    auto module = createBasicModule();
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function that simulates PostgreSQL runtime calls
    auto funcType = builder->getFunctionType({}, {});
    auto pgFunc = builder->create<func::FuncOp>(loc, "test_pg_calls", funcType);
    auto* pgBlock = pgFunc.addEntryBlock();
    
    builder->setInsertionPointToEnd(pgBlock);
    
    // Create mock PostgreSQL function declarations
    auto i32Type = builder->getI32Type();
    auto i64Type = builder->getI64Type();
    
    // Mock runtime function types
    auto readNextFuncType = builder->getFunctionType({i64Type}, i64Type);
    auto getIntFieldFuncType = builder->getFunctionType({i64Type, i32Type}, i32Type);
    auto storeResultFuncType = builder->getFunctionType({i32Type}, {});
    
    // Create function declarations
    auto readNextFunc = builder->create<func::FuncOp>(loc, "read_next_tuple_from_table", readNextFuncType);
    readNextFunc.setPrivate();
    
    auto getIntFieldFunc = builder->create<func::FuncOp>(loc, "get_int_field", getIntFieldFuncType); 
    getIntFieldFunc.setPrivate();
    
    auto storeResultFunc = builder->create<func::FuncOp>(loc, "store_int_result", storeResultFuncType);
    storeResultFunc.setPrivate();
    
    // Create runtime calls
    auto nullHandle = builder->create<arith::ConstantIntOp>(loc, 0, 64);
    auto readCall = builder->create<func::CallOp>(loc, readNextFunc, ValueRange{nullHandle});
    
    auto fieldIndex = builder->create<arith::ConstantIntOp>(loc, 0, 32);
    auto getFieldCall = builder->create<func::CallOp>(loc, getIntFieldFunc, 
        ValueRange{readCall.getResult(0), fieldIndex});
    
    auto storeCall = builder->create<func::CallOp>(loc, storeResultFunc, 
        ValueRange{getFieldCall.getResult(0)});
    
    // Add terminator
    builder->create<func::ReturnOp>(loc);
    
    // Verify structure
    EXPECT_TRUE(pgBlock->getTerminator());
    EXPECT_TRUE(succeeded(verify(module)));
    
    // Run lowering
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result));
    
    // Verify still valid
    EXPECT_TRUE(succeeded(verify(module)))
        << "PostgreSQL runtime call pattern should remain valid";
}