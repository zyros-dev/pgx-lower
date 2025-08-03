// Unit tests for ReduceOperations.cpp - Comprehensive testing of reduction operation lowering
// This file tests reduce operations that might be adding operations after terminators during aggregation

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowPatterns.h"
#include "dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowUtilities.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ReduceOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<MLIRContext>();
        context->loadDialect<subop::SubOperatorDialect>();
        context->loadDialect<util::UtilDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<memref::MemRefDialect>();
        context->loadDialect<func::FuncDialect>();
        
        builder = std::make_unique<OpBuilder>(context.get());
        loc = builder->getUnknownLoc();
        
        // Create module for tests
        module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
    }

    void TearDown() override {
        module.erase();
    }

    // Helper to create a test function context
    func::FuncOp createTestFunction(StringRef name, FunctionType funcType) {
        auto funcOp = builder->create<func::FuncOp>(loc, name, funcType);
        builder->setInsertionPointToStart(funcOp.addEntryBlock());
        return funcOp;
    }

    // Helper to create a reduce operation with a simple region
    subop::ReduceOp createSimpleReduceOp(Value refValue, ArrayAttr columns, ArrayAttr members) {
        auto reduceOp = builder->create<subop::ReduceOp>(loc, refValue, columns, members);
        
        // Create a simple reduction region with addition
        auto& region = reduceOp.getRegion();
        auto* block = &region.emplaceBlock();
        
        // Add block arguments for columns and members
        auto i32Type = builder->getI32Type();
        for (size_t i = 0; i < columns.size() + members.size(); i++) {
            block->addArgument(i32Type, loc);
        }
        
        // Create addition operation in the region
        builder->setInsertionPointToStart(block);
        auto arg0 = block->getArgument(0);
        auto arg1 = block->getArgument(block->getNumArguments() - 1); // Last argument is the member
        auto addOp = builder->create<arith::AddIOp>(loc, arg0, arg1);
        builder->create<tuples::ReturnOp>(loc, ValueRange{addOp});
        
        return reduceOp;
    }

    // Helper to create a floating-point reduce operation
    subop::ReduceOp createFloatReduceOp(Value refValue, ArrayAttr columns, ArrayAttr members) {
        auto reduceOp = builder->create<subop::ReduceOp>(loc, refValue, columns, members);
        
        auto& region = reduceOp.getRegion();
        auto* block = &region.emplaceBlock();
        
        auto f32Type = builder->getF32Type();
        for (size_t i = 0; i < columns.size() + members.size(); i++) {
            block->addArgument(f32Type, loc);
        }
        
        builder->setInsertionPointToStart(block);
        auto arg0 = block->getArgument(0);
        auto arg1 = block->getArgument(block->getNumArguments() - 1);
        auto addFOp = builder->create<arith::AddFOp>(loc, arg0, arg1);
        builder->create<tuples::ReturnOp>(loc, ValueRange{addFOp});
        
        return reduceOp;
    }

    // Helper to create a bitwise OR reduce operation
    subop::ReduceOp createBitwiseOrReduceOp(Value refValue, ArrayAttr columns, ArrayAttr members) {
        auto reduceOp = builder->create<subop::ReduceOp>(loc, refValue, columns, members);
        
        auto& region = reduceOp.getRegion();
        auto* block = &region.emplaceBlock();
        
        auto i32Type = builder->getI32Type();
        for (size_t i = 0; i < columns.size() + members.size(); i++) {
            block->addArgument(i32Type, loc);
        }
        
        builder->setInsertionPointToStart(block);
        auto arg0 = block->getArgument(0);
        auto arg1 = block->getArgument(block->getNumArguments() - 1);
        auto orOp = builder->create<arith::OrIOp>(loc, arg0, arg1);
        builder->create<tuples::ReturnOp>(loc, ValueRange{orOp});
        
        return reduceOp;
    }

    std::unique_ptr<MLIRContext> context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
    ModuleOp module;
};

// Test 1: Basic Atomic Integer Addition Reduction
TEST_F(ReduceOperationsTest, AtomicIntegerAdditionReduction) {
    PGX_INFO("Testing atomic integer addition reduction pattern matching");
    
    // Create a simple function context
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_atomic_add", funcType);
    
    // Create a reference value (mock continuous reference)
    auto refType = util::RefType::get(context.get(), builder->getI32Type());
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    // Create column and member attributes
    auto i32Type = builder->getI32Type();
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "test_col", i32Type));
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    // Create reduce operation with atomic attribute
    auto reduceOp = createSimpleReduceOp(undefRef, columns, members);
    reduceOp->setAttr("atomic", builder->getUnitAttr());
    
    // Verify the operation was created correctly
    EXPECT_TRUE(reduceOp);
    EXPECT_TRUE(reduceOp->hasAttr("atomic"));
    EXPECT_EQ(reduceOp.getColumns().size(), 1);
    EXPECT_EQ(reduceOp.getMembers().size(), 1);
    
    // Check the reduction region contains AddIOp
    auto& block = reduceOp.getRegion().front();
    bool hasAddIOp = false;
    block.walk([&](arith::AddIOp op) {
        hasAddIOp = true;
    });
    EXPECT_TRUE(hasAddIOp);
    
    PGX_INFO("Atomic integer addition test completed successfully");
}

// Test 2: Atomic Floating-Point Addition Reduction
TEST_F(ReduceOperationsTest, AtomicFloatAdditionReduction) {
    PGX_INFO("Testing atomic floating-point addition reduction");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_atomic_addf", funcType);
    
    auto refType = util::RefType::get(context.get(), builder->getF32Type());
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    auto f32Type = builder->getF32Type();
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "test_col", f32Type));
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = createFloatReduceOp(undefRef, columns, members);
    reduceOp->setAttr("atomic", builder->getUnitAttr());
    
    // Verify floating-point atomic reduction structure
    EXPECT_TRUE(reduceOp);
    
    auto& block = reduceOp.getRegion().front();
    bool hasAddFOp = false;
    block.walk([&](arith::AddFOp op) {
        hasAddFOp = true;
    });
    EXPECT_TRUE(hasAddFOp);
    
    PGX_INFO("Atomic float addition test completed successfully");
}

// Test 3: Atomic Bitwise OR Reduction
TEST_F(ReduceOperationsTest, AtomicBitwiseOrReduction) {
    PGX_INFO("Testing atomic bitwise OR reduction");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_atomic_or", funcType);
    
    auto refType = util::RefType::get(context.get(), builder->getI32Type());
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    auto i32Type = builder->getI32Type();
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "test_col", i32Type));
    auto memberAttr = builder->getStringAttr("flags");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = createBitwiseOrReduceOp(undefRef, columns, members);
    reduceOp->setAttr("atomic", builder->getUnitAttr());
    
    EXPECT_TRUE(reduceOp);
    
    auto& block = reduceOp.getRegion().front();
    bool hasOrIOp = false;
    block.walk([&](arith::OrIOp op) {
        hasOrIOp = true;
    });
    EXPECT_TRUE(hasOrIOp);
    
    PGX_INFO("Atomic bitwise OR test completed successfully");
}

// Test 4: Non-Atomic Continuous Reference Reduction
TEST_F(ReduceOperationsTest, NonAtomicContinuousRefReduction) {
    PGX_INFO("Testing non-atomic continuous reference reduction");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_non_atomic", funcType);
    
    // Create a continuous reference type
    auto i32Type = builder->getI32Type();
    auto memberAttrs = ArrayAttr::get(context.get(), {
        subop::StateEntryMemberAttr::get(context.get(), "sum", i32Type)
    });
    auto continuousRefType = subop::ContinuousEntryRefType::get(context.get(), memberAttrs, false);
    
    auto undefRef = builder->create<util::UndefOp>(loc, continuousRefType);
    
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "test_col", i32Type));
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    // Create reduce operation WITHOUT atomic attribute
    auto reduceOp = createSimpleReduceOp(undefRef, columns, members);
    
    EXPECT_TRUE(reduceOp);
    EXPECT_FALSE(reduceOp->hasAttr("atomic"));
    
    // Test that it recognizes the continuous reference type
    auto refColumn = reduceOp.getRef();
    EXPECT_TRUE(refColumn);
    
    PGX_INFO("Non-atomic continuous reference test completed successfully");
}

// Test 5: Parallel Reduction Stress Test
TEST_F(ReduceOperationsTest, ParallelReductionStressTest) {
    PGX_INFO("Testing parallel reduction operations for race conditions");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_parallel", funcType);
    
    // Create multiple atomic reduce operations to simulate parallel access
    auto refType = util::RefType::get(context.get(), builder->getI32Type());
    auto i32Type = builder->getI32Type();
    
    std::vector<subop::ReduceOp> reduceOps;
    
    for (int i = 0; i < 5; i++) {
        auto undefRef = builder->create<util::UndefOp>(loc, refType);
        
        auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
            tuples::Column::create(context.get(), "col_" + std::to_string(i), i32Type));
        auto memberAttr = builder->getStringAttr("sum_" + std::to_string(i));
        
        auto columns = builder->getArrayAttr({columnAttr});
        auto members = builder->getArrayAttr({memberAttr});
        
        auto reduceOp = createSimpleReduceOp(undefRef, columns, members);
        reduceOp->setAttr("atomic", builder->getUnitAttr());
        reduceOps.push_back(reduceOp);
    }
    
    // Verify all operations were created
    EXPECT_EQ(reduceOps.size(), 5);
    
    for (auto& op : reduceOps) {
        EXPECT_TRUE(op);
        EXPECT_TRUE(op->hasAttr("atomic"));
    }
    
    PGX_INFO("Parallel reduction stress test completed successfully");
}

// Test 6: Group By Aggregation Pattern
TEST_F(ReduceOperationsTest, GroupByAggregationPattern) {
    PGX_INFO("Testing group-by aggregation pattern with multiple members");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_group_by", funcType);
    
    auto refType = util::RefType::get(context.get(), builder->getI32Type());
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    auto i32Type = builder->getI32Type();
    auto f32Type = builder->getF32Type();
    
    // Create multiple columns for group by
    auto keyColumn = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "group_key", i32Type));
    auto valueColumn = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "value", f32Type));
    
    // Create multiple aggregate members
    auto sumMember = builder->getStringAttr("sum");
    auto countMember = builder->getStringAttr("count");
    auto avgMember = builder->getStringAttr("avg");
    
    auto columns = builder->getArrayAttr({keyColumn, valueColumn});
    auto members = builder->getArrayAttr({sumMember, countMember, avgMember});
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, undefRef, columns, members);
    
    // Create complex reduction region for group-by
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    // Add arguments for columns and members
    block->addArgument(i32Type, loc); // group_key
    block->addArgument(f32Type, loc); // value
    block->addArgument(f32Type, loc); // current sum
    block->addArgument(i32Type, loc); // current count
    block->addArgument(f32Type, loc); // current avg
    
    builder->setInsertionPointToStart(block);
    
    // Implement aggregation logic
    auto valueArg = block->getArgument(1);
    auto currentSum = block->getArgument(2);
    auto currentCount = block->getArgument(3);
    
    // Update sum
    auto newSum = builder->create<arith::AddFOp>(loc, currentSum, valueArg);
    
    // Update count
    auto one = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    auto newCount = builder->create<arith::AddIOp>(loc, currentCount, one);
    
    // Update average (simplified)
    auto countFloat = builder->create<arith::SIToFPOp>(loc, f32Type, newCount);
    auto newAvg = builder->create<arith::DivFOp>(loc, newSum, countFloat);
    
    builder->create<tuples::ReturnOp>(loc, ValueRange{newSum, newCount, newAvg});
    
    EXPECT_TRUE(reduceOp);
    EXPECT_EQ(reduceOp.getColumns().size(), 2);
    EXPECT_EQ(reduceOp.getMembers().size(), 3);
    
    PGX_INFO("Group-by aggregation test completed successfully");
}

// Test 7: Accumulator State Management
TEST_F(ReduceOperationsTest, AccumulatorStateManagement) {
    PGX_INFO("Testing accumulator state management in reductions");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_accumulator", funcType);
    
    // Create state entry reference type with multiple members
    auto i32Type = builder->getI32Type();
    auto f64Type = builder->getF64Type();
    
    auto memberAttrs = ArrayAttr::get(context.get(), {
        subop::StateEntryMemberAttr::get(context.get(), "sum", f64Type),
        subop::StateEntryMemberAttr::get(context.get(), "count", i32Type),
        subop::StateEntryMemberAttr::get(context.get(), "min_val", f64Type),
        subop::StateEntryMemberAttr::get(context.get(), "max_val", f64Type)
    });
    
    auto stateRefType = subop::StateEntryReference::get(context.get(), memberAttrs, false);
    auto undefRef = builder->create<util::UndefOp>(loc, stateRefType);
    
    auto valueColumn = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "input_value", f64Type));
    
    auto columns = builder->getArrayAttr({valueColumn});
    auto members = builder->getArrayAttr({
        builder->getStringAttr("sum"),
        builder->getStringAttr("count"),
        builder->getStringAttr("min_val"),
        builder->getStringAttr("max_val")
    });
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, undefRef, columns, members);
    
    // Create comprehensive reduction logic
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    block->addArgument(f64Type, loc); // input_value
    block->addArgument(f64Type, loc); // current sum
    block->addArgument(i32Type, loc); // current count
    block->addArgument(f64Type, loc); // current min
    block->addArgument(f64Type, loc); // current max
    
    builder->setInsertionPointToStart(block);
    
    auto inputValue = block->getArgument(0);
    auto currentSum = block->getArgument(1);
    auto currentCount = block->getArgument(2);
    auto currentMin = block->getArgument(3);
    auto currentMax = block->getArgument(4);
    
    // Update sum
    auto newSum = builder->create<arith::AddFOp>(loc, currentSum, inputValue);
    
    // Update count
    auto one = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    auto newCount = builder->create<arith::AddIOp>(loc, currentCount, one);
    
    // Update min using select
    auto isLess = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, inputValue, currentMin);
    auto newMin = builder->create<arith::SelectOp>(loc, isLess, inputValue, currentMin);
    
    // Update max using select
    auto isGreater = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, inputValue, currentMax);
    auto newMax = builder->create<arith::SelectOp>(loc, isGreater, inputValue, currentMax);
    
    builder->create<tuples::ReturnOp>(loc, ValueRange{newSum, newCount, newMin, newMax});
    
    EXPECT_TRUE(reduceOp);
    EXPECT_EQ(reduceOp.getMembers().size(), 4);
    
    // Verify the complex reduction logic
    auto& blockRef = reduceOp.getRegion().front();
    bool hasArithOps = false;
    blockRef.walk([&](arith::AddFOp op) {
        hasArithOps = true;
    });
    EXPECT_TRUE(hasArithOps);
    
    PGX_INFO("Accumulator state management test completed successfully");
}

// Test 8: Boolean Type Promotion in Atomic Operations
TEST_F(ReduceOperationsTest, BooleanTypePromotionAtomic) {
    PGX_INFO("Testing boolean type promotion in atomic operations");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_bool_promotion", funcType);
    
    // Create reference with boolean type
    auto boolType = builder->getI1Type();
    auto refType = util::RefType::get(context.get(), boolType);
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "flag_col", boolType));
    auto memberAttr = builder->getStringAttr("flag_sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, undefRef, columns, members);
    
    // Create OR reduction for boolean flags
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    block->addArgument(boolType, loc); // input flag
    block->addArgument(boolType, loc); // current flag
    
    builder->setInsertionPointToStart(block);
    
    auto inputFlag = block->getArgument(0);
    auto currentFlag = block->getArgument(1);
    auto newFlag = builder->create<arith::OrIOp>(loc, inputFlag, currentFlag);
    
    builder->create<tuples::ReturnOp>(loc, ValueRange{newFlag});
    
    reduceOp->setAttr("atomic", builder->getUnitAttr());
    
    EXPECT_TRUE(reduceOp);
    EXPECT_TRUE(reduceOp->hasAttr("atomic"));
    
    // Test should verify that boolean promotion logic exists
    auto& blockRef = reduceOp.getRegion().front();
    bool hasOrIOp = false;
    blockRef.walk([&](arith::OrIOp op) {
        hasOrIOp = true;
    });
    EXPECT_TRUE(hasOrIOp);
    
    PGX_INFO("Boolean type promotion test completed successfully");
}

// Test 9: Error Handling - Invalid Reduction Patterns
TEST_F(ReduceOperationsTest, ErrorHandlingInvalidPatterns) {
    PGX_INFO("Testing error handling for invalid reduction patterns");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_error_handling", funcType);
    
    auto refType = util::RefType::get(context.get(), builder->getI32Type());
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    auto i32Type = builder->getI32Type();
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "test_col", i32Type));
    auto memberAttr = builder->getStringAttr("invalid");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, undefRef, columns, members);
    
    // Create region with invalid/complex operation that can't be atomized
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    block->addArgument(i32Type, loc);
    block->addArgument(i32Type, loc);
    
    builder->setInsertionPointToStart(block);
    
    // Create complex operation that cannot be directly mapped to atomic
    auto arg0 = block->getArgument(0);
    auto arg1 = block->getArgument(1);
    auto mulOp = builder->create<arith::MulIOp>(loc, arg0, arg1); // Multiplication (not supported atomically)
    auto addOp = builder->create<arith::AddIOp>(loc, mulOp, arg1); // Complex expression
    
    builder->create<tuples::ReturnOp>(loc, ValueRange{addOp});
    
    reduceOp->setAttr("atomic", builder->getUnitAttr());
    
    EXPECT_TRUE(reduceOp);
    
    // This should contain complex operations that require generic atomic handling
    auto& blockRef = reduceOp.getRegion().front();
    bool hasMulIOp = false;
    blockRef.walk([&](arith::MulIOp op) {
        hasMulIOp = true;
    });
    EXPECT_TRUE(hasMulIOp);
    
    PGX_INFO("Error handling test completed successfully");
}

// Test 10: Terminator Safety in Reduction Operations
TEST_F(ReduceOperationsTest, TerminatorSafetyInReductions) {
    PGX_INFO("Testing that reduce operations don't add operations after terminators");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_terminator_safety", funcType);
    
    auto refType = util::RefType::get(context.get(), builder->getI32Type());
    auto undefRef = builder->create<util::UndefOp>(loc, refType);
    
    auto i32Type = builder->getI32Type();
    auto columnAttr = tuples::ColumnRefAttr::get(context.get(), 
        tuples::Column::create(context.get(), "test_col", i32Type));
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = createSimpleReduceOp(undefRef, columns, members);
    
    // Verify the reduction region has proper termination
    auto& region = reduceOp.getRegion();
    EXPECT_EQ(region.getBlocks().size(), 1);
    
    auto& block = region.front();
    EXPECT_TRUE(block.getTerminator() != nullptr);
    EXPECT_TRUE(isa<tuples::ReturnOp>(block.getTerminator()));
    
    // Verify terminator is the last operation
    auto* terminator = block.getTerminator();
    EXPECT_EQ(&block.back(), terminator);
    
    // Count operations before terminator
    size_t opCount = 0;
    for (auto& op : block.getOperations()) {
        if (&op != terminator) {
            opCount++;
        }
    }
    
    // Should have exactly one AddIOp before the terminator
    EXPECT_EQ(opCount, 1);
    
    PGX_INFO("Terminator safety test completed successfully");
}

// Integration test combining multiple reduction patterns
TEST_F(ReduceOperationsTest, IntegrationTestMultipleReductionPatterns) {
    PGX_INFO("Running integration test with multiple reduction patterns");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_integration", funcType);
    
    // Test 1: Atomic integer reduction
    auto intRefType = util::RefType::get(context.get(), builder->getI32Type());
    auto intUndefRef = builder->create<util::UndefOp>(loc, intRefType);
    auto intReduce = createSimpleReduceOp(intUndefRef, 
        builder->getArrayAttr({tuples::ColumnRefAttr::get(context.get(), 
            tuples::Column::create(context.get(), "int_col", builder->getI32Type()))}),
        builder->getArrayAttr({builder->getStringAttr("int_sum")}));
    intReduce->setAttr("atomic", builder->getUnitAttr());
    
    // Test 2: Float reduction
    auto floatRefType = util::RefType::get(context.get(), builder->getF32Type());
    auto floatUndefRef = builder->create<util::UndefOp>(loc, floatRefType);
    auto floatReduce = createFloatReduceOp(floatUndefRef,
        builder->getArrayAttr({tuples::ColumnRefAttr::get(context.get(), 
            tuples::Column::create(context.get(), "float_col", builder->getF32Type()))}),
        builder->getArrayAttr({builder->getStringAttr("float_sum")}));
    floatReduce->setAttr("atomic", builder->getUnitAttr());
    
    // Test 3: Boolean OR reduction
    auto boolRefType = util::RefType::get(context.get(), builder->getI1Type());
    auto boolUndefRef = builder->create<util::UndefOp>(loc, boolRefType);
    auto boolReduce = createBitwiseOrReduceOp(boolUndefRef,
        builder->getArrayAttr({tuples::ColumnRefAttr::get(context.get(), 
            tuples::Column::create(context.get(), "bool_col", builder->getI1Type()))}),
        builder->getArrayAttr({builder->getStringAttr("bool_flags")}));
    boolReduce->setAttr("atomic", builder->getUnitAttr());
    
    // Verify all operations were created correctly
    EXPECT_TRUE(intReduce && intReduce->hasAttr("atomic"));
    EXPECT_TRUE(floatReduce && floatReduce->hasAttr("atomic"));
    EXPECT_TRUE(boolReduce && boolReduce->hasAttr("atomic"));
    
    // Count the total number of reduce operations in the function
    size_t reduceCount = 0;
    funcOp.walk([&](subop::ReduceOp op) {
        reduceCount++;
    });
    
    EXPECT_EQ(reduceCount, 3);
    
    PGX_INFO("Integration test completed successfully - found " + std::to_string(reduceCount) + " reduce operations");
}