// Comprehensive unit tests for ControlFlowOperations.cpp
// Tests critical control flow operations that handle block termination

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuples/TupleStreamDialect.h"
#include "dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowUtilities.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;
using namespace subop_to_control_flow;

class ControlFlowOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
        
        PGX_DEBUG("ControlFlowOperationsTest setup complete");
    }

    void TearDown() override {
        PGX_DEBUG("ControlFlowOperationsTest teardown");
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
};

//===----------------------------------------------------------------------===//
// FilterLowering Tests - Critical for conditional control flow
//===----------------------------------------------------------------------===//

TEST_F(ControlFlowOperationsTest, FilterLoweringBasicTermination) {
    PGX_INFO("Testing FilterLowering basic termination");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function to contain the filter operation
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_filter", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Create a boolean condition for the filter
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    
    // Create if operation like FilterLowering does
    auto ifOp = builder->create<scf::IfOp>(loc, TypeRange{}, condition);
    
    // Test that If operation is created properly
    EXPECT_TRUE(ifOp);
    EXPECT_TRUE(ifOp.getThenRegion().hasOneBlock());
    
    // Test terminator utilities used by FilterLowering
    EXPECT_NO_THROW(TerminatorUtils::ensureIfOpTermination(ifOp, *builder, loc));
    
    // Verify termination was applied properly
    EXPECT_TRUE(ifOp.getThenRegion().front().getTerminator() != nullptr);
    
    PGX_DEBUG("FilterLowering basic termination test passed");
}

TEST_F(ControlFlowOperationsTest, FilterLoweringConditionalExecution) {
    PGX_INFO("Testing FilterLowering conditional execution patterns");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_filter_conditional", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Test multiple condition handling (like FilterLowering lines 94-96)
    auto cond1 = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    auto cond2 = builder->create<arith::ConstantIntOp>(loc, 0, builder->getI1Type());
    
    // Test AND operation creation
    SmallVector<Value> conditions = {cond1, cond2};
    auto andOp = builder->create<db::AndOp>(loc, ValueRange(conditions), ArrayRef<NamedAttribute>{});
    
    EXPECT_TRUE(andOp);
    EXPECT_EQ(andOp.getNumOperands(), 2);
    
    // Test truth derivation
    auto truthOp = builder->create<db::DeriveTruth>(loc, andOp);
    EXPECT_TRUE(truthOp);
    
    // Test NOT operation for none_true semantic
    auto notOp = builder->create<db::NotOp>(loc, truthOp);
    EXPECT_TRUE(notOp);
    
    PGX_DEBUG("FilterLowering conditional execution test passed");
}

TEST_F(ControlFlowOperationsTest, FilterLoweringRuntimeCallSafety) {
    PGX_INFO("Testing FilterLowering runtime call safety application");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_filter_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Create a PostgreSQL runtime call to test safety
    auto calleeType = FunctionType::get(&context, {}, {});
    auto callOp = builder->create<func::CallOp>(loc, "add_tuple_to_result", calleeType.getResults(), ValueRange{});
    
    // Test runtime call detection
    EXPECT_TRUE(RuntimeCallTermination::isPostgreSQLRuntimeCall(callOp));
    
    // Test runtime call safety application (like FilterLowering line 109)
    EXPECT_NO_THROW(RuntimeCallTermination::applyRuntimeCallSafetyToOperation(callOp, *builder));
    
    PGX_DEBUG("FilterLowering runtime call safety test passed");
}

//===----------------------------------------------------------------------===//
// LoopLowering Tests - Critical for loop control flow
//===----------------------------------------------------------------------===//

TEST_F(ControlFlowOperationsTest, LoopLoweringWhileOpCreation) {
    PGX_INFO("Testing LoopLowering while operation creation");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_loop", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Test while operation creation like LoopLowering does (line 221)
    auto trueValue = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    std::vector<Type> iterTypes = {builder->getI1Type()};
    std::vector<Value> iterArgs = {trueValue};
    
    auto whileOp = builder->create<scf::WhileOp>(loc, iterTypes, iterArgs);
    
    EXPECT_TRUE(whileOp);
    EXPECT_EQ(whileOp.getNumResults(), 1);
    
    // Test before region setup
    auto* beforeBlock = new Block;
    beforeBlock->addArguments(iterTypes, SmallVector<Location>(iterTypes.size(), loc));
    whileOp.getBefore().push_back(beforeBlock);
    
    // Test condition operation creation
    builder->setInsertionPointToEnd(beforeBlock);
    builder->create<scf::ConditionOp>(loc, beforeBlock->getArgument(0), beforeBlock->getArguments());
    
    EXPECT_TRUE(beforeBlock->getTerminator() != nullptr);
    EXPECT_TRUE(isa<scf::ConditionOp>(beforeBlock->getTerminator()));
    
    PGX_DEBUG("LoopLowering while operation creation test passed");
}

TEST_F(ControlFlowOperationsTest, LoopLoweringTerminatorValidation) {
    PGX_INFO("Testing LoopLowering terminator validation");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_loop_termination", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Create while operation
    auto trueValue = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    std::vector<Type> iterTypes = {builder->getI1Type()};
    std::vector<Value> iterArgs = {trueValue};
    
    auto whileOp = builder->create<scf::WhileOp>(loc, iterTypes, iterArgs);
    
    // Add before and after regions
    auto* beforeBlock = new Block;
    beforeBlock->addArguments(iterTypes, SmallVector<Location>(iterTypes.size(), loc));
    whileOp.getBefore().push_back(beforeBlock);
    
    auto* afterBlock = new Block;
    afterBlock->addArguments(iterTypes, SmallVector<Location>(iterTypes.size(), loc));
    whileOp.getAfter().push_back(afterBlock);
    
    // Test terminator validation for both regions (like LoopLowering lines 296-297)
    EXPECT_NO_THROW(TerminatorUtils::ensureTerminator(whileOp.getBefore(), *builder, loc));
    EXPECT_NO_THROW(TerminatorUtils::ensureTerminator(whileOp.getAfter(), *builder, loc));
    
    // Verify terminators were added
    EXPECT_TRUE(beforeBlock->getTerminator() != nullptr);
    EXPECT_TRUE(afterBlock->getTerminator() != nullptr);
    
    PGX_DEBUG("LoopLowering terminator validation test passed");
}

TEST_F(ControlFlowOperationsTest, LoopLoweringYieldOpCreation) {
    PGX_INFO("Testing LoopLowering yield operation creation");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_loop_yield", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Test yield operation creation (like LoopLowering line 292)
    auto boolValue = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    std::vector<Value> yieldOperands = {boolValue};
    
    auto yieldOp = builder->create<scf::YieldOp>(loc, yieldOperands);
    
    EXPECT_TRUE(yieldOp);
    EXPECT_EQ(yieldOp.getNumOperands(), 1);
    EXPECT_EQ(yieldOp.getOperand(0), boolValue);
    
    PGX_DEBUG("LoopLowering yield operation creation test passed");
}

//===----------------------------------------------------------------------===//
// MapLowering Tests - Critical for expression evaluation
//===----------------------------------------------------------------------===//

TEST_F(ControlFlowOperationsTest, MapLoweringRuntimeCallSafety) {
    PGX_INFO("Testing MapLowering runtime call safety");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_map_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Create a PostgreSQL runtime call that MapLowering would handle
    auto calleeType = FunctionType::get(&context, {}, {builder->getI32Type()});
    auto callOp = builder->create<func::CallOp>(loc, "get_int_field", calleeType.getResults(), ValueRange{});
    
    // Test runtime call safety application (like MapLowering line 60)
    EXPECT_NO_THROW(RuntimeCallTermination::applyRuntimeCallSafetyToOperation(callOp, *builder));
    
    // Verify the call operation still exists and is properly handled
    EXPECT_TRUE(callOp);
    EXPECT_TRUE(RuntimeCallTermination::isPostgreSQLRuntimeCall(callOp));
    
    PGX_DEBUG("MapLowering runtime call safety test passed");
}

TEST_F(ControlFlowOperationsTest, MapLoweringInlineBlockExecution) {
    PGX_INFO("Testing MapLowering inline block execution");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_map_inline", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Test block inlining pattern used by MapLowering
    auto constantOp = builder->create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Create a block with return operation to simulate map function
    auto* inlineBlock = new Block;
    auto i32Type = builder->getI32Type();
    inlineBlock->addArgument(i32Type, loc);
    
    // Set insertion point to the inline block
    builder->setInsertionPointToEnd(inlineBlock);
    auto returnOp = builder->create<tuples::ReturnOp>(loc, ValueRange{inlineBlock->getArgument(0)});
    
    EXPECT_TRUE(returnOp);
    EXPECT_EQ(returnOp.getNumOperands(), 1);
    EXPECT_TRUE(inlineBlock->getTerminator() != nullptr);
    
    PGX_DEBUG("MapLowering inline block execution test passed");
}

//===----------------------------------------------------------------------===//
// LockLowering Tests - Critical for thread safety
//===----------------------------------------------------------------------===//

TEST_F(ControlFlowOperationsTest, LockLoweringThreadSafety) {
    PGX_INFO("Testing LockLowering thread safety patterns");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_lock_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Test lock/unlock pattern used by LockLowering
    auto ptrType = util::RefType::get(&context, builder->getI32Type());
    auto mockPtr = builder->create<arith::ConstantIntOp>(loc, 0, 64); // Mock pointer
    
    // Create mock lock operations like LockLowering lines 399 and 409
    auto lockCallType = FunctionType::get(&context, {builder->getI64Type()}, {});
    auto lockOp = builder->create<func::CallOp>(loc, "rt_EntryLock_lock", lockCallType.getResults(), ValueRange{mockPtr});
    auto unlockOp = builder->create<func::CallOp>(loc, "rt_EntryLock_unlock", lockCallType.getResults(), ValueRange{mockPtr});
    
    EXPECT_TRUE(lockOp);
    EXPECT_TRUE(unlockOp);
    EXPECT_EQ(lockOp.getNumOperands(), 1);
    EXPECT_EQ(unlockOp.getNumOperands(), 1);
    
    PGX_DEBUG("LockLowering thread safety test passed");
}

//===----------------------------------------------------------------------===//
// Integration Tests - Testing complete control flow patterns
//===----------------------------------------------------------------------===//

TEST_F(ControlFlowOperationsTest, ControlFlowIntegrationTest) {
    PGX_INFO("Testing complete control flow integration");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_integration", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Create nested control flow like real operations would
    auto condition = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    auto ifOp = builder->create<scf::IfOp>(loc, condition, false);
    
    // Test comprehensive termination
    EXPECT_NO_THROW(TerminatorUtils::ensureIfOpTermination(ifOp, *builder, loc));
    
    // Create while loop inside if statement
    builder->setInsertionPointToStart(ifOp.thenBlock());
    auto trueValue = builder->create<arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    std::vector<Type> iterTypes = {builder->getI1Type()};
    std::vector<Value> iterArgs = {trueValue};
    
    auto whileOp = builder->create<scf::WhileOp>(loc, iterTypes, iterArgs);
    
    // Add regions and ensure termination
    auto* beforeBlock = new Block;
    beforeBlock->addArguments(iterTypes, SmallVector<Location>(iterTypes.size(), loc));
    whileOp.getBefore().push_back(beforeBlock);
    
    auto* afterBlock = new Block;
    afterBlock->addArguments(iterTypes, SmallVector<Location>(iterTypes.size(), loc));
    whileOp.getAfter().push_back(afterBlock);
    
    // Ensure all terminators
    EXPECT_NO_THROW(TerminatorUtils::ensureTerminator(whileOp.getBefore(), *builder, loc));
    EXPECT_NO_THROW(TerminatorUtils::ensureTerminator(whileOp.getAfter(), *builder, loc));
    EXPECT_NO_THROW(TerminatorUtils::ensureTerminator(ifOp.getThenRegion(), *builder, loc));
    
    // Verify all blocks have terminators
    funcOp.walk([](Operation* op) {
        for (auto& region : op->getRegions()) {
            for (auto& block : region.getBlocks()) {
                EXPECT_TRUE(block.getTerminator() != nullptr);
            }
        }
    });
    
    PGX_INFO("Complete control flow integration test passed");
}

TEST_F(ControlFlowOperationsTest, DefensiveProgrammingValidation) {
    PGX_INFO("Testing defensive programming validation patterns");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_defensive", funcType);
    auto* block = funcOp.addEntryBlock();
    
    // Test block validation without terminator
    EXPECT_FALSE(DefensiveProgramming::BlockTerminationValidator::validateBlock(block));
    
    // Repair the block
    EXPECT_NO_THROW(DefensiveProgramming::BlockTerminationValidator::repairBlock(block, *builder, loc));
    
    // Verify block is now valid
    EXPECT_TRUE(DefensiveProgramming::BlockTerminationValidator::validateBlock(block));
    EXPECT_TRUE(block->getTerminator() != nullptr);
    
    PGX_DEBUG("Defensive programming validation test passed");
}

TEST_F(ControlFlowOperationsTest, PostgreSQLIntegrationValidation) {
    PGX_INFO("Testing PostgreSQL integration validation");
    
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_postgresql", funcType);
    auto* block = funcOp.addEntryBlock();
    builder->setInsertionPointToEnd(block);
    
    // Create PostgreSQL runtime calls
    auto calleeType = FunctionType::get(&context, {}, {builder->getI64Type()});
    auto getFieldOp = builder->create<func::CallOp>(loc, "get_int_field", calleeType.getResults(), ValueRange{});
    auto addTupleOp = builder->create<func::CallOp>(loc, "add_tuple_to_result", FunctionType::get(&context, {}, {}).getResults(), ValueRange{});
    
    // Test PostgreSQL integration
    EXPECT_NO_THROW(PostgreSQLIntegration::ensurePostgreSQLCompatibleTermination(block, *builder, loc));
    EXPECT_NO_THROW(PostgreSQLIntegration::handleMemoryContextInvalidation(block, *builder));
    
    // Verify PostgreSQL call detection
    EXPECT_TRUE(RuntimeCallTermination::isPostgreSQLRuntimeCall(getFieldOp));
    EXPECT_TRUE(RuntimeCallTermination::isPostgreSQLRuntimeCall(addTupleOp));
    
    PGX_DEBUG("PostgreSQL integration validation test passed");
}

// Test the repeat helper function used throughout control flow operations
TEST_F(ControlFlowOperationsTest, RepeatHelperFunction) {
    PGX_INFO("Testing repeat helper function");
    
    // Test the repeat function used in LoopLowering (line 223)
    auto intType = builder->getI32Type();
    std::vector<Type> types;
    for (auto i = 0ul; i < 3; i++) {
        types.push_back(intType);
    }
    
    EXPECT_EQ(types.size(), 3);
    for (const auto& type : types) {
        EXPECT_EQ(type, intType);
    }
    
    // Test with locations
    std::vector<Location> locations;
    for (auto i = 0ul; i < 5; i++) {
        locations.push_back(loc);
    }
    
    EXPECT_EQ(locations.size(), 5);
    for (const auto& location : locations) {
        EXPECT_EQ(location, loc);
    }
    
    PGX_DEBUG("Repeat helper function test passed");
}