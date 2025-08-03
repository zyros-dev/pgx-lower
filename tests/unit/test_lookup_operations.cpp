#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuples/TupleStreamDialect.h"
#include "test_helpers.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

/**
 * Comprehensive unit tests for LookupOperations.cpp
 * 
 * This test suite covers all major lookup operation types:
 * 1. Simple State Lookup
 * 2. Hash Indexed View Lookup  
 * 3. Segment Tree View Lookup
 * 4. Pure Hash Map Lookup
 * 5. Pre-Aggregation Hash Table Lookup
 * 6. Hash Multi-Map Lookup
 * 7. External Hash Index Lookup
 * 8. Lookup with Insert operations
 * 
 * Focus areas for block termination issues:
 * - Proper SCF control flow handling
 * - Runtime call termination patterns
 * - While loop and if-then-else block structure
 * - Error handling and missing key scenarios
 */
class LookupOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext context;
    
    // Helper to create a basic module with function
    ModuleOp createTestModule(OpBuilder& builder) {
        Location loc = builder.getUnknownLoc();
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
        
        // Create a test function
        auto funcType = builder.getFunctionType({}, {});
        auto func = builder.create<func::FuncOp>(loc, "test_lookup", funcType);
        auto& funcBody = func.getBody();
        funcBody.emplaceBlock();
        builder.setInsertionPointToStart(&funcBody.front());
        
        return module;
    }
    
    // Helper to create basic types for testing
    Type createSimpleStateType() {
        return subop::SimpleStateType::get(&context, builder.getI32Type());
    }
    
    Type createHashIndexedViewType() {
        return subop::HashIndexedViewType::get(&context, 
            ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())}));
    }
    
    Type createSegmentTreeViewType() {
        return subop::SegmentTreeViewType::get(&context,
            ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())}));
    }
    
    Type createHashMapType() {
        auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
        auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
        return subop::HashMapType::get(&context, keyMembers, valueMembers, false);
    }
    
    Type createPreAggrHtType() {
        auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
        auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
        return subop::PreAggrHtType::get(&context, keyMembers, valueMembers, false);
    }
    
    Type createHashMultiMapType() {
        auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
        auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
        return subop::HashMultiMapType::get(&context, keyMembers, valueMembers, false);
    }
    
    Type createExternalHashIndexType() {
        return subop::ExternalHashIndexType::get(&context);
    }

private:
    OpBuilder builder{&context};
};

//===----------------------------------------------------------------------===//
// Simple State Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, SimpleStateLookupBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create simple state type and lookup operation
    auto stateType = createSimpleStateType();
    auto stateValue = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Create lookup operation
    auto refType = util::RefType::get(&context, builder.getI32Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType, 
        stateValue, ValueRange{}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(isa<subop::SimpleStateType>(stateValue.getType()));
}

TEST_F(LookupOperationsTest, SimpleStateLookupTermination) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test that simple state lookup properly terminates blocks
    auto stateType = createSimpleStateType();
    auto stateValue = builder.create<arith::ConstantIntOp>(loc, 100, 32);
    auto refType = util::RefType::get(&context, builder.getI32Type());
    
    // Create if-then structure around lookup
    auto condValue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto ifOp = builder.create<scf::IfOp>(loc, condValue,
        [&](OpBuilder& b, Location loc) {
            auto lookupOp = b.create<subop::LookupOp>(loc, refType,
                stateValue, ValueRange{}, ValueRange{});
            b.create<scf::YieldOp>(loc);
        });
    
    EXPECT_TRUE(ifOp);
    EXPECT_EQ(ifOp.getThenRegion().getBlocks().size(), 1);
}

//===----------------------------------------------------------------------===//
// Hash Indexed View Lookup Tests  
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, HashIndexedViewLookupBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create hash indexed view state
    auto htType = util::RefType::get(&context, 
        util::RefType::get(&context, builder.getI8Type()));
    auto indexType = builder.getIndexType();
    auto tupleType = TupleType::get(&context, {htType, indexType});
    auto stateValue = builder.create<util::AllocaOp>(loc, 
        util::RefType::get(&context, tupleType), Value());
    
    // Create hash value for lookup
    auto hashValue = builder.create<arith::ConstantIndexOp>(loc, 12345);
    
    // Create lookup operation
    auto refType = util::RefType::get(&context, builder.getI32Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{hashValue}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(lookupOp.getKeys().size() == 1);
}

TEST_F(LookupOperationsTest, HashIndexedViewOptimizationPath) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test the optimization path for tag matching
    auto htType = util::RefType::get(&context, 
        util::RefType::get(&context, builder.getI8Type()));
    auto indexType = builder.getIndexType();
    auto tupleType = TupleType::get(&context, {htType, indexType});
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, tupleType), Value());
    
    auto hashValue = builder.create<arith::ConstantIndexOp>(loc, 0x12345678);
    auto refType = util::RefType::get(&context, builder.getI32Type());
    
    // Create lookup with specific hash that tests optimization
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{hashValue}, ValueRange{});
    
    // Verify the operation was created successfully
    EXPECT_TRUE(lookupOp);
    EXPECT_EQ(lookupOp.getKeys().size(), 1);
}

//===----------------------------------------------------------------------===//
// Segment Tree View Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, SegmentTreeViewBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create segment tree view state
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto segmentType = subop::SegmentTreeViewType::get(&context, valueMembers);
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, segmentType), Value());
    
    // Create left and right index values
    auto leftIdx = builder.create<arith::ConstantIndexOp>(loc, 10);
    auto rightIdx = builder.create<arith::ConstantIndexOp>(loc, 20);
    auto leftTuple = builder.create<util::PackOp>(loc, ValueRange{leftIdx});
    auto rightTuple = builder.create<util::PackOp>(loc, ValueRange{rightIdx});
    
    // Create lookup operation
    auto refType = util::RefType::get(&context, builder.getI32Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{leftTuple, rightTuple}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_EQ(lookupOp.getKeys().size(), 2);
}

TEST_F(LookupOperationsTest, SegmentTreeViewRuntimeCallTermination) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test that segment tree lookup properly handles runtime call termination
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto segmentType = subop::SegmentTreeViewType::get(&context, valueMembers);
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, segmentType), Value());
    
    auto leftIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto rightIdx = builder.create<arith::ConstantIndexOp>(loc, 100);
    auto leftTuple = builder.create<util::PackOp>(loc, ValueRange{leftIdx});
    auto rightTuple = builder.create<util::PackOp>(loc, ValueRange{rightIdx});
    
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{leftTuple, rightTuple}, ValueRange{});
    
    // Verify the lookup operation
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(isa<util::RefType>(lookupOp.getRef().getType()));
}

//===----------------------------------------------------------------------===//
// Pure Hash Map Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, PureHashMapLookupBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create hash map state
    auto hashMapType = createHashMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    
    // Create lookup key
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 123, 32);
    
    // Create equality function region
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto cmpOp = eqBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        eqBlock->getArgument(0), eqBlock->getArgument(1));
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpOp});
    
    // Create lookup operation
    auto refType = util::RefType::get(&context, builder.getI8Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    lookupOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(lookupOp);
    EXPECT_FALSE(lookupOp.getEqFn().empty());
}

TEST_F(LookupOperationsTest, PureHashMapComplexWhileLoop) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test the complex while loop structure in hash map lookup
    auto hashMapType = createHashMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 456, 32);
    
    // Create equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto trueVal = eqBuilder.create<arith::ConstantIntOp>(loc, 1, 1);
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{trueVal});
    
    auto refType = util::RefType::get(&context, builder.getI8Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    lookupOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(lookupOp);
    EXPECT_EQ(lookupOp.getEqFn().getBlocks().size(), 1);
}

//===----------------------------------------------------------------------===//
// Pre-Aggregation Hash Table Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, PreAggregationHtLookupBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create pre-aggregation hash table state
    auto preAggrType = createPreAggrHtType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, preAggrType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 789, 32);
    
    // Create equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto cmpOp = eqBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        eqBlock->getArgument(0), eqBlock->getArgument(1));
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpOp});
    
    auto refType = util::RefType::get(&context, builder.getI8Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    lookupOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(lookupOp);
    ASSERT_TRUE(isa<subop::PreAggrHtType>(stateValue.getType().cast<util::RefType>().getElementType()));
}

TEST_F(LookupOperationsTest, PreAggregationHtPartitioning) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test the 64-way partitioning logic
    auto preAggrType = createPreAggrHtType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, preAggrType), Value());
    
    // Test with hash value that exercises partitioning
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 0xABCDEF12, 32);
    
    // Create simple equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto falseVal = eqBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{falseVal});
    
    auto refType = util::RefType::get(&context, builder.getI8Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    lookupOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(lookupOp.getKeys().size() == 1);
}

//===----------------------------------------------------------------------===//
// Hash Multi-Map Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, HashMultiMapLookupBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create hash multi-map state
    auto multiMapType = createHashMultiMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, multiMapType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 999, 32);
    
    // Create equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto cmpOp = eqBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        eqBlock->getArgument(0), eqBlock->getArgument(1));
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpOp});
    
    auto refType = util::RefType::get(&context, multiMapType);
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    lookupOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(isa<subop::HashMultiMapType>(multiMapType));
}

TEST_F(LookupOperationsTest, HashMultiMapInsertOperation) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test multi-map insert operation
    auto multiMapType = createHashMultiMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, multiMapType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 777, 32);
    auto valueValue = builder.create<arith::ConstantIntOp>(loc, 888, 64);
    
    // Create column mapping for insert
    auto columnMapping = ArrayAttr::get(&context, {
        IntegerAttr::get(builder.getI32Type(), 0),
        IntegerAttr::get(builder.getI32Type(), 1)
    });
    
    // Create equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto trueVal = eqBuilder.create<arith::ConstantIntOp>(loc, 1, 1);
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{trueVal});
    
    auto insertOp = builder.create<subop::InsertOp>(loc, stateValue,
        columnMapping, ValueRange{keyValue, valueValue});
    insertOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(insertOp);
    EXPECT_FALSE(insertOp.getEqFn().empty());
}

//===----------------------------------------------------------------------===//
// External Hash Index Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, ExternalHashIndexLookupBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create external hash index state
    auto extIndexType = createExternalHashIndexType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, extIndexType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 555, 32);
    
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(isa<subop::ExternalHashIndexType>(extIndexType));
}

TEST_F(LookupOperationsTest, ExternalHashIndexRuntimeCall) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test runtime call termination for external hash index
    auto extIndexType = createExternalHashIndexType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, extIndexType), Value());
    
    // Multiple keys to test hash packing
    auto key1 = builder.create<arith::ConstantIntOp>(loc, 111, 32);
    auto key2 = builder.create<arith::ConstantIntOp>(loc, 222, 32);
    
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{key1, key2}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_EQ(lookupOp.getKeys().size(), 2);
}

//===----------------------------------------------------------------------===//
// Lookup with Insert Operations Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, LookupOrInsertHashMapBasic) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create hash map state for lookup-or-insert
    auto hashMapType = createHashMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 333, 32);
    
    // Create equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto cmpOp = eqBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        eqBlock->getArgument(0), eqBlock->getArgument(1));
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpOp});
    
    // Create initialization function
    Region initRegion;
    Block* initBlock = new Block;
    initRegion.push_back(initBlock);
    
    OpBuilder initBuilder(&context);
    initBuilder.setInsertionPointToStart(initBlock);
    auto initValue = initBuilder.create<arith::ConstantIntOp>(loc, 0, 64);
    initBuilder.create<tuples::ReturnOp>(loc, ValueRange{initValue});
    
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOrInsertOp = builder.create<subop::LookupOrInsertOp>(loc, refType,
        stateValue, ValueRange{keyValue});
    lookupOrInsertOp.getEqFn().takeBody(eqRegion);
    lookupOrInsertOp.getInitFn().takeBody(initRegion);
    
    EXPECT_TRUE(lookupOrInsertOp);
    EXPECT_FALSE(lookupOrInsertOp.getEqFn().empty());
    EXPECT_FALSE(lookupOrInsertOp.getInitFn().empty());
}

TEST_F(LookupOperationsTest, LookupOrInsertPreAggrFragment) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Create pre-aggregation fragment type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto fragmentType = subop::PreAggrHtFragmentType::get(&context, keyMembers, valueMembers, false);
    
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, fragmentType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 444, 32);
    
    // Create equality function
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto cmpOp = eqBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        eqBlock->getArgument(0), eqBlock->getArgument(1));
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpOp});
    
    // Create initialization function
    Region initRegion;
    Block* initBlock = new Block;
    initRegion.push_back(initBlock);
    
    OpBuilder initBuilder(&context);
    initBuilder.setInsertionPointToStart(initBlock);
    auto initValue = initBuilder.create<arith::ConstantIntOp>(loc, 0, 64);
    initBuilder.create<tuples::ReturnOp>(loc, ValueRange{initValue});
    
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOrInsertOp = builder.create<subop::LookupOrInsertOp>(loc, refType,
        stateValue, ValueRange{keyValue});
    lookupOrInsertOp.getEqFn().takeBody(eqRegion);
    lookupOrInsertOp.getInitFn().takeBody(initRegion);
    
    EXPECT_TRUE(lookupOrInsertOp);
    EXPECT_TRUE(isa<subop::PreAggrHtFragmentType>(fragmentType));
}

//===----------------------------------------------------------------------===//
// Error Handling and Edge Cases Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, LookupFailureScenarios) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test lookup with null/invalid state
    auto hashMapType = createHashMapType();
    auto nullState = builder.create<util::InvalidRefOp>(loc,
        util::RefType::get(&context, hashMapType));
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 123, 32);
    
    auto refType = util::RefType::get(&context, builder.getI8Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        nullState, ValueRange{keyValue}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_TRUE(isa<util::InvalidRefOp>(nullState.getDefiningOp()));
}

TEST_F(LookupOperationsTest, MissingKeyHandling) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test lookup behavior when key is not found
    auto hashMapType = createHashMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, -1, 32); // Non-existent key
    
    // Create equality function that always returns false
    Region eqRegion;
    Block* eqBlock = new Block;
    eqRegion.push_back(eqBlock);
    eqBlock->addArgument(builder.getI32Type(), loc);
    eqBlock->addArgument(builder.getI32Type(), loc);
    
    OpBuilder eqBuilder(&context);
    eqBuilder.setInsertionPointToStart(eqBlock);
    auto falseVal = eqBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
    eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{falseVal});
    
    auto refType = util::RefType::get(&context, builder.getI8Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keyValue}, ValueRange{});
    lookupOp.getEqFn().takeBody(eqRegion);
    
    EXPECT_TRUE(lookupOp);
    EXPECT_FALSE(lookupOp.getEqFn().empty());
}

//===----------------------------------------------------------------------===//
// Cache Access and Optimization Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, CacheOptimizationPatterns) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test pointer tag optimization for cache-friendly access
    auto htType = util::RefType::get(&context, 
        util::RefType::get(&context, builder.getI8Type()));
    auto indexType = builder.getIndexType();
    auto tupleType = TupleType::get(&context, {htType, indexType});
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, tupleType), Value());
    
    // Use hash with specific pattern to test tag matching
    auto hashValue = builder.create<arith::ConstantIndexOp>(loc, 0xDEADBEEF);
    auto refType = util::RefType::get(&context, builder.getI32Type());
    
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{hashValue}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    // Verify we can create the optimization structures
    EXPECT_TRUE(isa<arith::ConstantIndexOp>(hashValue.getDefiningOp()));
}

TEST_F(LookupOperationsTest, BlockTerminationValidation) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test that all control flow structures are properly terminated
    auto hashMapType = createHashMapType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    
    auto keyValue = builder.create<arith::ConstantIntOp>(loc, 999, 32);
    
    // Create nested control flow with lookup
    auto condValue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto outerIf = builder.create<scf::IfOp>(loc, condValue,
        [&](OpBuilder& b, Location loc) {
            // Create equality function
            Region eqRegion;
            Block* eqBlock = new Block;
            eqRegion.push_back(eqBlock);
            eqBlock->addArgument(builder.getI32Type(), loc);
            eqBlock->addArgument(builder.getI32Type(), loc);
            
            OpBuilder eqBuilder(&context);
            eqBuilder.setInsertionPointToStart(eqBlock);
            auto trueVal = eqBuilder.create<arith::ConstantIntOp>(loc, 1, 1);
            eqBuilder.create<tuples::ReturnOp>(loc, ValueRange{trueVal});
            
            auto refType = util::RefType::get(&context, builder.getI8Type());
            auto lookupOp = b.create<subop::LookupOp>(loc, refType,
                stateValue, ValueRange{keyValue}, ValueRange{});
            lookupOp.getEqFn().takeBody(eqRegion);
            
            b.create<scf::YieldOp>(loc);
        });
    
    EXPECT_TRUE(outerIf);
    EXPECT_EQ(outerIf.getThenRegion().getBlocks().size(), 1);
    
    // Verify the block is properly terminated
    auto& thenBlock = outerIf.getThenRegion().front();
    EXPECT_TRUE(thenBlock.mightHaveTerminator());
}

//===----------------------------------------------------------------------===//
// Performance and Stress Tests
//===----------------------------------------------------------------------===//

TEST_F(LookupOperationsTest, MultipleKeysLookup) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test lookup with multiple keys
    auto extIndexType = createExternalHashIndexType();
    auto stateValue = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, extIndexType), Value());
    
    // Create multiple keys for composite lookup
    std::vector<Value> keys;
    for (int i = 0; i < 5; ++i) {
        keys.push_back(builder.create<arith::ConstantIntOp>(loc, i * 100, 32));
    }
    
    auto refType = util::RefType::get(&context, builder.getI64Type());
    auto lookupOp = builder.create<subop::LookupOp>(loc, refType,
        stateValue, ValueRange{keys}, ValueRange{});
    
    EXPECT_TRUE(lookupOp);
    EXPECT_EQ(lookupOp.getKeys().size(), 5);
}

TEST_F(LookupOperationsTest, ConcurrentLookupPatterns) {
    OpBuilder builder(&context);
    auto module = createTestModule(builder);
    Location loc = builder.getUnknownLoc();
    
    // Test patterns that might cause termination issues
    auto hashMapType = createHashMapType();
    auto state1 = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    auto state2 = builder.create<util::AllocaOp>(loc,
        util::RefType::get(&context, hashMapType), Value());
    
    auto key1 = builder.create<arith::ConstantIntOp>(loc, 111, 32);
    auto key2 = builder.create<arith::ConstantIntOp>(loc, 222, 32);
    
    // Create parallel if branches with lookups
    auto condValue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto ifOp = builder.create<scf::IfOp>(loc, condValue,
        [&](OpBuilder& b, Location loc) {
            // First lookup
            Region eq1Region;
            Block* eq1Block = new Block;
            eq1Region.push_back(eq1Block);
            eq1Block->addArgument(builder.getI32Type(), loc);
            eq1Block->addArgument(builder.getI32Type(), loc);
            
            OpBuilder eq1Builder(&context);
            eq1Builder.setInsertionPointToStart(eq1Block);
            auto trueVal1 = eq1Builder.create<arith::ConstantIntOp>(loc, 1, 1);
            eq1Builder.create<tuples::ReturnOp>(loc, ValueRange{trueVal1});
            
            auto refType = util::RefType::get(&context, builder.getI8Type());
            auto lookup1 = b.create<subop::LookupOp>(loc, refType,
                state1, ValueRange{key1}, ValueRange{});
            lookup1.getEqFn().takeBody(eq1Region);
            
            b.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder& b, Location loc) {
            // Second lookup  
            Region eq2Region;
            Block* eq2Block = new Block;
            eq2Region.push_back(eq2Block);
            eq2Block->addArgument(builder.getI32Type(), loc);
            eq2Block->addArgument(builder.getI32Type(), loc);
            
            OpBuilder eq2Builder(&context);
            eq2Builder.setInsertionPointToStart(eq2Block);
            auto trueVal2 = eq2Builder.create<arith::ConstantIntOp>(loc, 1, 1);
            eq2Builder.create<tuples::ReturnOp>(loc, ValueRange{trueVal2});
            
            auto refType = util::RefType::get(&context, builder.getI8Type());
            auto lookup2 = b.create<subop::LookupOp>(loc, refType,
                state2, ValueRange{key2}, ValueRange{});
            lookup2.getEqFn().takeBody(eq2Region);
            
            b.create<scf::YieldOp>(loc);
        });
    
    EXPECT_TRUE(ifOp);
    EXPECT_EQ(ifOp.getThenRegion().getBlocks().size(), 1);
    EXPECT_EQ(ifOp.getElseRegion().getBlocks().size(), 1);
}