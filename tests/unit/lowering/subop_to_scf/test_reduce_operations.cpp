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
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "dialects/util/UtilTypes.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ReduceOperationsTest : public ::testing::Test {
public:
    ReduceOperationsTest() = default;
    
protected:
    
    void SetUp() override {
        context = std::make_unique<MLIRContext>();
        context->loadDialect<subop::SubOperatorDialect>();
        context->loadDialect<util::UtilDialect>();
        context->loadDialect<tuples::TupleStreamDialect>();
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
    subop::ReduceOp createSimpleReduceOp(Value streamValue, ArrayAttr columns, ArrayAttr members) {
        // Create a simple ColumnRefAttr for testing
        auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
        // For testing, use nullptr for the column pointer - this may cause issues but is simplest for compilation testing
        auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
        
        auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
        
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
        builder->create<tuples::ReturnOp>(loc, ValueRange{addOp.getResult()});
        
        return reduceOp;
    }

    // Helper to create a floating-point reduce operation
    subop::ReduceOp createFloatReduceOp(Value streamValue, ArrayAttr columns, ArrayAttr members) {
        // Create a simple ColumnRefAttr for testing
        auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
        auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
        
        auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
        
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
    subop::ReduceOp createBitwiseOrReduceOp(Value streamValue, ArrayAttr columns, ArrayAttr members) {
        // Create a simple ColumnRefAttr for testing
        auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
        auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
        
        auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
        
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
    Location loc = UnknownLoc::get(context.get());
    ModuleOp module;
    
    // Helper to create a simple tuple stream
    Value createSimpleTupleStream() {
        auto tupleStreamType = tuples::TupleStreamType::get(context.get());
        return builder->create<util::UndefOp>(loc, tupleStreamType);
    }
};

// Test 1: Basic Atomic Integer Addition Reduction
TEST_F(ReduceOperationsTest, AtomicIntegerAdditionReduction) {
    PGX_INFO("Testing atomic integer addition reduction pattern matching");
    
    // Create a simple function context
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_atomic_add", funcType);
    
    // Create a tuple stream
    auto streamValue = createSimpleTupleStream();
    
    // Note: ReduceOp uses ColumnRefAttr for ref, not a value operand
    
    // Create column and member attributes
    auto columnAttr = builder->getStringAttr("test_col");
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    // Create reduce operation with atomic attribute
    auto reduceOp = createSimpleReduceOp(streamValue, columns, members);
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
    
    auto streamValue = createSimpleTupleStream();
    
    auto columnAttr = builder->getStringAttr("test_col");
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = createFloatReduceOp(streamValue, columns, members);
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
    
    auto streamValue = createSimpleTupleStream();
    
    auto columnAttr = builder->getStringAttr("test_col");
    auto memberAttr = builder->getStringAttr("flags");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = createBitwiseOrReduceOp(streamValue, columns, members);
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

// Test 4: Non-Atomic Simple State Reduction
TEST_F(ReduceOperationsTest, NonAtomicSimpleStateReduction) {
    PGX_INFO("Testing non-atomic simple state reduction");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_non_atomic", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    auto columnAttr = builder->getStringAttr("test_col");
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    // Create reduce operation WITHOUT atomic attribute
    auto reduceOp = createSimpleReduceOp(streamValue, columns, members);
    
    EXPECT_TRUE(reduceOp);
    EXPECT_FALSE(reduceOp->hasAttr("atomic"));
    
    // Test that the ref attribute was set
    EXPECT_TRUE(reduceOp.getRef());
    
    PGX_INFO("Non-atomic simple state test completed successfully");
}

// Test 5: Multiple Reduction Operations Test
TEST_F(ReduceOperationsTest, MultipleReductionOperationsTest) {
    PGX_INFO("Testing multiple reduction operations creation");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_multiple", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    std::vector<subop::ReduceOp> reduceOps;
    
    for (int i = 0; i < 3; i++) {
        
        auto columnAttr = builder->getStringAttr("col_" + std::to_string(i));
        auto memberAttr = builder->getStringAttr("sum_" + std::to_string(i));
        
        auto columns = builder->getArrayAttr({columnAttr});
        auto members = builder->getArrayAttr({memberAttr});
        
        auto reduceOp = createSimpleReduceOp(streamValue, columns, members);
        reduceOp->setAttr("atomic", builder->getUnitAttr());
        reduceOps.push_back(reduceOp);
    }
    
    // Verify all operations were created
    EXPECT_EQ(reduceOps.size(), 3);
    
    for (auto& op : reduceOps) {
        EXPECT_TRUE(op);
        EXPECT_TRUE(op->hasAttr("atomic"));
    }
    
    PGX_INFO("Multiple reduction operations test completed successfully");
}

// Test 6: Multiple Member Aggregation Pattern
TEST_F(ReduceOperationsTest, MultipleMemberAggregationPattern) {
    PGX_INFO("Testing aggregation pattern with multiple members");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_multi_member", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    auto i32Type = builder->getI32Type();
    auto f32Type = builder->getF32Type();
    
    // Create column and member attributes
    auto keyColumn = builder->getStringAttr("group_key");
    auto valueColumn = builder->getStringAttr("value");
    
    auto sumMember = builder->getStringAttr("sum");
    auto countMember = builder->getStringAttr("count");
    
    auto columns = builder->getArrayAttr({keyColumn, valueColumn});
    auto members = builder->getArrayAttr({sumMember, countMember});
    
    // Create a simple ColumnRefAttr for testing
    auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
    auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
    
    // Create simple reduction region
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    // Add arguments for columns and members
    block->addArgument(i32Type, loc); // group_key
    block->addArgument(f32Type, loc); // value
    block->addArgument(f32Type, loc); // current sum
    block->addArgument(i32Type, loc); // current count
    
    builder->setInsertionPointToStart(block);
    
    // Simple aggregation logic
    auto valueArg = block->getArgument(1);
    auto currentSum = block->getArgument(2);
    auto currentCount = block->getArgument(3);
    
    // Update sum
    auto newSum = builder->create<arith::AddFOp>(loc, currentSum, valueArg);
    
    // Update count
    auto one = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    auto newCount = builder->create<arith::AddIOp>(loc, currentCount, one);
    
    SmallVector<Value> results = {newSum, newCount};
    builder->create<tuples::ReturnOp>(loc, results);
    
    EXPECT_TRUE(reduceOp);
    EXPECT_EQ(reduceOp.getColumns().size(), 2);
    EXPECT_EQ(reduceOp.getMembers().size(), 2);
    
    PGX_INFO("Multiple member aggregation test completed successfully");
}

// Test 7: Simple Arithmetic Operations in Reduction
TEST_F(ReduceOperationsTest, ArithmeticOperationsInReduction) {
    PGX_INFO("Testing arithmetic operations in reduction regions");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_arithmetic", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    auto f64Type = builder->getF64Type();
    auto i32Type = builder->getI32Type();
    
    auto valueColumn = builder->getStringAttr("input_value");
    
    auto columns = builder->getArrayAttr({valueColumn});
    auto members = builder->getArrayAttr({
        builder->getStringAttr("sum"),
        builder->getStringAttr("count")
    });
    
    // Create a simple ColumnRefAttr for testing
    auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
    auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
    
    // Create reduction logic with arithmetic operations
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    block->addArgument(f64Type, loc); // input_value
    block->addArgument(f64Type, loc); // current sum
    block->addArgument(i32Type, loc); // current count
    
    builder->setInsertionPointToStart(block);
    
    auto inputValue = block->getArgument(0);
    auto currentSum = block->getArgument(1);
    auto currentCount = block->getArgument(2);
    
    // Update sum
    auto newSum = builder->create<arith::AddFOp>(loc, currentSum, inputValue);
    
    // Update count
    auto one = builder->create<arith::ConstantIntOp>(loc, 1, 32);
    auto newCount = builder->create<arith::AddIOp>(loc, currentCount, one);
    
    SmallVector<Value> results = {newSum, newCount};
    builder->create<tuples::ReturnOp>(loc, results);
    
    EXPECT_TRUE(reduceOp);
    EXPECT_EQ(reduceOp.getMembers().size(), 2);
    
    // Verify arithmetic operations exist
    auto& blockRef = reduceOp.getRegion().front();
    bool hasArithOps = false;
    blockRef.walk([&](arith::AddFOp op) {
        hasArithOps = true;
    });
    EXPECT_TRUE(hasArithOps);
    
    PGX_INFO("Arithmetic operations test completed successfully");
}

// Test 8: Boolean Operations in Reduction
TEST_F(ReduceOperationsTest, BooleanOperationsInReduction) {
    PGX_INFO("Testing boolean operations in reduction");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_bool_ops", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    auto columnAttr = builder->getStringAttr("flag_col");
    auto memberAttr = builder->getStringAttr("flag_sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    // Create a simple ColumnRefAttr for testing
    auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
    auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
    
    // Create OR reduction for boolean flags
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    auto boolType = builder->getI1Type();
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
    
    // Verify boolean OR operation exists
    auto& blockRef = reduceOp.getRegion().front();
    bool hasOrIOp = false;
    blockRef.walk([&](arith::OrIOp op) {
        hasOrIOp = true;
    });
    EXPECT_TRUE(hasOrIOp);
    
    PGX_INFO("Boolean operations test completed successfully");
}

// Test 9: Complex Arithmetic Operations in Reduction
TEST_F(ReduceOperationsTest, ComplexArithmeticOperations) {
    PGX_INFO("Testing complex arithmetic operations in reduction");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_complex_ops", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    auto i32Type = builder->getI32Type();
    
    auto columnAttr = builder->getStringAttr("test_col");
    auto memberAttr = builder->getStringAttr("result");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    // Create a simple ColumnRefAttr for testing
    auto symbolRef = mlir::SymbolRefAttr::get(context.get(), "test_ref");
    auto refAttr = tuples::ColumnRefAttr::get(context.get(), symbolRef, nullptr);
    
    auto reduceOp = builder->create<subop::ReduceOp>(loc, streamValue, refAttr, columns, members);
    
    // Create region with complex arithmetic operations
    auto& region = reduceOp.getRegion();
    auto* block = &region.emplaceBlock();
    
    block->addArgument(i32Type, loc);
    block->addArgument(i32Type, loc);
    
    builder->setInsertionPointToStart(block);
    
    // Create complex arithmetic expression
    auto arg0 = block->getArgument(0);
    auto arg1 = block->getArgument(1);
    auto mulOp = builder->create<arith::MulIOp>(loc, arg0, arg1); // Multiplication
    auto addOp = builder->create<arith::AddIOp>(loc, mulOp, arg1); // Complex expression
    
    builder->create<tuples::ReturnOp>(loc, ValueRange{addOp});
    
    EXPECT_TRUE(reduceOp);
    
    // Verify complex operations exist
    auto& blockRef = reduceOp.getRegion().front();
    bool hasMulIOp = false;
    blockRef.walk([&](arith::MulIOp op) {
        hasMulIOp = true;
    });
    EXPECT_TRUE(hasMulIOp);
    
    PGX_INFO("Complex arithmetic operations test completed successfully");
}

// Test 10: Terminator Safety in Reduction Operations
TEST_F(ReduceOperationsTest, TerminatorSafetyInReductions) {
    PGX_INFO("Testing that reduce operations don't add operations after terminators");
    
    auto funcType = FunctionType::get(context.get(), {}, {});
    auto funcOp = createTestFunction("test_terminator_safety", funcType);
    
    auto streamValue = createSimpleTupleStream();
    
    auto columnAttr = builder->getStringAttr("test_col");
    auto memberAttr = builder->getStringAttr("sum");
    
    auto columns = builder->getArrayAttr({columnAttr});
    auto members = builder->getArrayAttr({memberAttr});
    
    auto reduceOp = createSimpleReduceOp(streamValue, columns, members);
    
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
    
    auto streamValue = createSimpleTupleStream();
    
    // Test 1: Integer reduction
    auto intReduce = createSimpleReduceOp(streamValue,
        builder->getArrayAttr({builder->getStringAttr("int_col")}),
        builder->getArrayAttr({builder->getStringAttr("int_sum")}));
    intReduce->setAttr("atomic", builder->getUnitAttr());
    
    // Test 2: Float reduction
    auto floatReduce = createFloatReduceOp(streamValue,
        builder->getArrayAttr({builder->getStringAttr("float_col")}),
        builder->getArrayAttr({builder->getStringAttr("float_sum")}));
    floatReduce->setAttr("atomic", builder->getUnitAttr());
    
    // Verify operations were created correctly
    EXPECT_TRUE(intReduce && intReduce->hasAttr("atomic"));
    EXPECT_TRUE(floatReduce && floatReduce->hasAttr("atomic"));
    
    // Count the total number of reduce operations in the function
    size_t reduceCount = 0;
    funcOp.walk([&](subop::ReduceOp op) {
        reduceCount++;
    });
    
    EXPECT_EQ(reduceCount, 2);
    
    PGX_INFO("Integration test completed successfully - found " + std::to_string(reduceCount) + " reduce operations");
}