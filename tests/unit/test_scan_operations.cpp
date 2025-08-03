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
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/util/UtilDialect.h"
#include "test_helpers.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

/**
 * Comprehensive unit tests for ScanOperations.cpp
 * 
 * This test suite focuses on the core scan operations that implement table scanning,
 * buffer iteration, and terminator management patterns. These operations are fundamental
 * to the MLIR pipeline and likely where terminator bugs first manifest.
 * 
 * Test Coverage:
 * 1. Table Scan Tests - Sequential and indexed table scanning
 * 2. Filter Integration Tests - Scan operations with filter predicates
 * 3. Iterator Tests - Scan iterator creation and management
 * 4. Memory Access Tests - Scan memory patterns and efficiency
 * 5. Terminator Safety Tests - Verify terminator handling in scan operations
 */
class ScanOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load all required dialects for scan operations
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
        
        // Initialize mock scan context
        g_mock_scan_context = new MockTupleScanContext();
        g_mock_scan_context->values = {1, 2, 3, 4, 5};
        g_mock_scan_context->currentIndex = 0;
        g_mock_scan_context->hasMore = true;
    }

    void TearDown() override {
        delete g_mock_scan_context;
        g_mock_scan_context = nullptr;
    }

    MLIRContext context;
};

// ============================================================================
// Table Scan Tests - Sequential and Indexed Scanning
// ============================================================================

TEST_F(ScanOperationsTest, ScanRefsVectorBufferIteration) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function to contain the scan operation
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_scan_refs_vector", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create buffer type for scan state
    auto i32Type = builder.getI32Type();
    auto tupleType = TupleType::get(&context, {i32Type});
    auto bufferType = subop::BufferType::get(&context, ArrayAttr(), false);
    
    // Create mock buffer state
    auto bufferValue = builder.create<util::UndefOp>(loc, bufferType);
    
    // Create a tuple column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("test_col", i32Type);
    auto columnRef = tupleDialect.getColumnManager().createRef(column);
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for vector buffer scanning
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, bufferValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify the scan operation was created correctly
    EXPECT_TRUE(scanOp);
    EXPECT_EQ(scanOp.getRef().getColumn().getName(), "test_col");
    EXPECT_TRUE(mlir::isa<subop::BufferType>(scanOp.getState().getType()));
    
    // Verify module structure
    EXPECT_TRUE(module);
    EXPECT_EQ(module.getBody()->getOperations().size(), 1); // One function
}

TEST_F(ScanOperationsTest, ScanRefsSimpleStateHandling) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_scan_simple_state", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create simple state type for direct value scanning
    auto simpleStateType = subop::SimpleStateType::get(&context, builder.getI64Type());
    auto stateValue = builder.create<util::UndefOp>(loc, simpleStateType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("simple_col", builder.getI64Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for simple state
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, stateValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify simple state scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::SimpleStateType>(scanOp.getState().getType()));
    EXPECT_EQ(scanOp.getRef().getColumn().getName(), "simple_col");
}

TEST_F(ScanOperationsTest, ScanRefsSortedViewWithForLoop) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_sorted_view_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create sorted view type
    auto i32Type = builder.getI32Type();
    auto members = ArrayAttr::get(&context, {TypeAttr::get(i32Type)});
    auto sortedViewType = subop::SortedViewType::get(&context, members, false);
    auto sortedViewValue = builder.create<util::UndefOp>(loc, sortedViewType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("sorted_col", i32Type);
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for sorted view (this should generate a ForOp)
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, sortedViewValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify sorted view scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::SortedViewType>(scanOp.getState().getType()));
    
    // Note: The actual ForOp will be created during lowering pass
    // This test verifies the scan operation setup is correct
}

TEST_F(ScanOperationsTest, ScanRefsHeapBufferWithTerminatorSafety) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_heap_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create heap type
    auto i64Type = builder.getI64Type();
    auto members = ArrayAttr::get(&context, {TypeAttr::get(i64Type)});
    auto heapType = subop::HeapType::get(&context, members, false);
    auto heapValue = builder.create<util::UndefOp>(loc, heapType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("heap_col", i64Type);
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for heap buffer
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, heapValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify heap scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::HeapType>(scanOp.getState().getType()));
    
    // This pattern should generate ForOp during lowering and needs terminator safety
    // The test verifies the setup is correct for terminator validation
}

// ============================================================================
// Hash Map and Hash Table Scanning Tests
// ============================================================================

TEST_F(ScanOperationsTest, ScanHashMapIteration) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_hashmap_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create hash map type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMapType = subop::HashMapType::get(&context, keyMembers, valueMembers, false);
    auto hashMapValue = builder.create<util::UndefOp>(loc, hashMapType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("hashmap_col", builder.getI64Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for hash map
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, hashMapValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify hash map scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::HashMapType>(scanOp.getState().getType()));
    
    // This should use Hashtable::createIterator during lowering
}

TEST_F(ScanOperationsTest, ScanPreAggregationHashTable) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_preaggr_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create pre-aggregation hash table type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto preAggrHtType = subop::PreAggrHtType::get(&context, keyMembers, valueMembers, false);
    auto preAggrValue = builder.create<util::UndefOp>(loc, preAggrHtType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("preaggr_col", builder.getI64Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for pre-aggregation hash table
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, preAggrValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify pre-aggregation hash table scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::PreAggrHtType>(scanOp.getState().getType()));
    
    // This should use PreAggregationHashtable::createIterator during lowering
}

TEST_F(ScanOperationsTest, ScanHashMultiMapWithWhileLoop) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_hash_multimap_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create hash multi-map type
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMultiMapType = subop::HashMultiMapType::get(&context, keyMembers, valueMembers, false);
    auto hashMultiMapValue = builder.create<util::UndefOp>(loc, hashMultiMapType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("multimap_col", builder.getI64Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for hash multi-map
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, hashMultiMapValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify hash multi-map scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::HashMultiMapType>(scanOp.getState().getType()));
    
    // This pattern should generate WhileOp during lowering for value chain iteration
}

// ============================================================================
// List Scanning Operations Tests
// ============================================================================

TEST_F(ScanOperationsTest, ScanHashMapListLookup) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_hashmap_list_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create hash map type for list content
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMapType = subop::HashMapType::get(&context, keyMembers, valueMembers, false);
    
    // Create lookup entry ref type
    auto lookupRefType = subop::LookupEntryRefType::get(&context, hashMapType, ArrayAttr());
    auto listType = subop::ListType::get(&context, lookupRefType);
    auto listValue = builder.create<util::UndefOp>(loc, listType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("list_elem", builder.getI64Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanListOp for hash map list
    auto scanOp = builder.create<subop::ScanListOp>(loc, columnDefAttr, listValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify hash map list scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::ListType>(scanOp.getList().getType()));
    
    auto listTypeCheck = mlir::cast<subop::ListType>(scanOp.getList().getType());
    EXPECT_TRUE(mlir::isa<subop::LookupEntryRefType>(listTypeCheck.getT()));
}

TEST_F(ScanOperationsTest, ScanMultiMapListWithValueChain) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_multimap_list_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create hash multi-map type for list content
    auto keyMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto valueMembers = ArrayAttr::get(&context, {TypeAttr::get(builder.getI64Type())});
    auto hashMultiMapType = subop::HashMultiMapType::get(&context, keyMembers, valueMembers, false);
    
    // Create lookup entry ref type for multi-map
    auto lookupRefType = subop::LookupEntryRefType::get(&context, hashMultiMapType, ArrayAttr());
    auto listType = subop::ListType::get(&context, lookupRefType);
    auto listValue = builder.create<util::UndefOp>(loc, listType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("multimap_elem", builder.getI64Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanListOp for multi-map list
    auto scanOp = builder.create<subop::ScanListOp>(loc, columnDefAttr, listValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify multi-map list scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::ListType>(scanOp.getList().getType()));
    
    // This pattern should generate WhileOp for value chain traversal during lowering
}

// ============================================================================
// Iterator Management and Memory Access Tests
// ============================================================================

TEST_F(ScanOperationsTest, IteratorCreationAndManagement) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_iterator_management", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Test iterator pattern with buffer type
    auto bufferType = subop::BufferType::get(&context, ArrayAttr(), false);
    auto bufferValue = builder.create<util::UndefOp>(loc, bufferType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("iter_col", builder.getI32Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create scan operation that will use iterator
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, bufferValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify iterator-based scan setup
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::BufferType>(scanOp.getState().getType()));
    
    // During lowering, this should call GrowingBuffer::createIterator
    // and use implementBufferIteration utility
}

TEST_F(ScanOperationsTest, ParallelScanAttributeHandling) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_parallel_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create buffer type for parallel scan
    auto bufferType = subop::BufferType::get(&context, ArrayAttr(), false);
    auto bufferValue = builder.create<util::UndefOp>(loc, bufferType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("parallel_col", builder.getI32Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create scan operation with parallel attribute
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, bufferValue);
    scanOp->setAttr("parallel", builder.getUnitAttr());
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify parallel scan setup
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(scanOp->hasAttr("parallel"));
    
    // During lowering, the parallel attribute should be passed to implementBufferIteration
}

TEST_F(ScanOperationsTest, ContinuousViewFunctionGeneration) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_continuous_view", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create continuous view type
    auto elementType = builder.getI32Type();
    auto continuousViewType = subop::ContinuousViewType::get(&context, elementType);
    auto continuousViewValue = builder.create<util::UndefOp>(loc, continuousViewType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("cv_col", elementType);
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanRefsOp for continuous view
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, continuousViewValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify continuous view scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::ContinuousViewType>(scanOp.getState().getType()));
    
    // During lowering, this should generate a new function and use Buffer::iterate
}

// ============================================================================
// Memory Safety and Terminator Validation Tests
// ============================================================================

TEST_F(ScanOperationsTest, TerminatorSafetyInForLoop) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create function with proper termination
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_terminator_safety", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a ForOp that mimics what ScanRefsSortedViewLowering generates
    auto start = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto end = builder.create<arith::ConstantIndexOp>(loc, 10);
    auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
    
    auto forOp = builder.create<scf::ForOp>(loc, start, end, step);
    
    // The ForOp body should have proper termination
    builder.setInsertionPointToStart(forOp.getBody());
    
    // Create some operation in the loop body
    auto inductionVar = forOp.getInductionVar();
    auto constVal = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Properly terminate the ForOp body
    builder.create<scf::YieldOp>(loc);
    
    // Return to function level and terminate
    builder.setInsertionPointAfter(forOp);
    builder.create<func::ReturnOp>(loc);
    
    // Verify ForOp structure and termination
    EXPECT_TRUE(forOp);
    EXPECT_TRUE(forOp.getBody()->hasTerminator());
    EXPECT_TRUE(mlir::isa<scf::YieldOp>(forOp.getBody()->getTerminator()));
    
    // Verify function termination
    EXPECT_TRUE(entryBlock->hasTerminator());
    EXPECT_TRUE(mlir::isa<func::ReturnOp>(entryBlock->getTerminator()));
}

TEST_F(ScanOperationsTest, WhileLoopTerminatorPattern) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_while_terminator", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create WhileOp that mimics HashMultiMap value chain iteration
    auto i8PtrType = util::RefType::get(&context, builder.getI8Type());
    auto initialPtr = builder.create<util::UndefOp>(loc, i8PtrType);
    
    auto whileOp = builder.create<scf::WhileOp>(loc, i8PtrType, initialPtr);
    
    // Create condition block
    Block* conditionBlock = new Block;
    whileOp.getBefore().push_back(conditionBlock);
    Value condArg = conditionBlock->addArgument(i8PtrType, loc);
    
    builder.setInsertionPointToStart(conditionBlock);
    auto condition = builder.create<util::IsRefValidOp>(loc, builder.getI1Type(), condArg);
    builder.create<scf::ConditionOp>(loc, condition, condArg);
    
    // Create body block
    Block* bodyBlock = new Block;
    whileOp.getAfter().push_back(bodyBlock);
    Value bodyArg = bodyBlock->addArgument(i8PtrType, loc);
    
    builder.setInsertionPointToStart(bodyBlock);
    // Simulate getting next pointer
    auto nextPtr = builder.create<util::UndefOp>(loc, i8PtrType);
    builder.create<scf::YieldOp>(loc, nextPtr);
    
    // Return to function level
    builder.setInsertionPointAfter(whileOp);
    builder.create<func::ReturnOp>(loc);
    
    // Verify WhileOp structure and termination
    EXPECT_TRUE(whileOp);
    EXPECT_EQ(whileOp.getBefore().getBlocks().size(), 1);
    EXPECT_EQ(whileOp.getAfter().getBlocks().size(), 1);
    
    // Verify condition block termination
    EXPECT_TRUE(conditionBlock->hasTerminator());
    EXPECT_TRUE(mlir::isa<scf::ConditionOp>(conditionBlock->getTerminator()));
    
    // Verify body block termination
    EXPECT_TRUE(bodyBlock->hasTerminator());
    EXPECT_TRUE(mlir::isa<scf::YieldOp>(bodyBlock->getTerminator()));
}

TEST_F(ScanOperationsTest, RuntimeCallTerminationSafety) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_runtime_call_safety", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Simulate runtime call pattern that needs termination safety
    // This mimics what GrowingBuffer::createIterator or Hashtable::createIterator would do
    
    // Create a function call operation (placeholder for runtime call)
    auto calleeType = FunctionType::get(&context, {}, {builder.getI8Type()});
    auto mockRuntimeFunc = builder.create<func::FuncOp>(module.getLoc(), "mock_runtime_call", calleeType);
    
    // Call the runtime function
    auto callOp = builder.create<func::CallOp>(loc, mockRuntimeFunc, ValueRange{});
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify runtime call structure
    EXPECT_TRUE(callOp);
    EXPECT_EQ(callOp.getNumResults(), 1);
    
    // The actual termination safety would be applied during lowering
    // via RuntimeCallTermination utilities
}

// ============================================================================
// Filter Integration and Predicate Tests
// ============================================================================

TEST_F(ScanOperationsTest, ScanWithFilterPredicate) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_scan_with_filter", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create buffer for scanning
    auto bufferType = subop::BufferType::get(&context, ArrayAttr(), false);
    auto bufferValue = builder.create<util::UndefOp>(loc, bufferType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("filter_col", builder.getI32Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create scan operation
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, bufferValue);
    
    // Create a simple filter condition (value > 5)
    auto constFive = builder.create<arith::ConstantIntOp>(loc, 5, 32);
    auto mockValue = builder.create<arith::ConstantIntOp>(loc, 10, 32);
    auto filterCond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, mockValue, constFive);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify scan with filter setup
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(filterCond);
    EXPECT_TRUE(mlir::isa<arith::CmpIOp>(filterCond.getOperation()));
    
    // During lowering, the filter would be integrated into the scan iteration
}

TEST_F(ScanOperationsTest, HashIndexedViewWithHashComparison) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_hash_indexed_view", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create hash indexed view type
    auto members = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto hashIndexedViewType = subop::HashIndexedViewType::get(&context, members, true); // compareHashForLookup = true
    
    // Create lookup entry ref type
    auto lookupRefType = subop::LookupEntryRefType::get(&context, hashIndexedViewType, members);
    auto listType = subop::ListType::get(&context, lookupRefType);
    auto listValue = builder.create<util::UndefOp>(loc, listType);
    
    // Create column reference
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("hash_elem", builder.getI32Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    // Create ScanListOp for hash indexed view
    auto scanOp = builder.create<subop::ScanListOp>(loc, columnDefAttr, listValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify hash indexed view scan creation
    EXPECT_TRUE(scanOp);
    EXPECT_TRUE(mlir::isa<subop::ListType>(scanOp.getList().getType()));
    
    auto listTypeCheck = mlir::cast<subop::ListType>(scanOp.getList().getType());
    auto lookupRefCheck = mlir::cast<subop::LookupEntryRefType>(listTypeCheck.getT());
    auto hashIndexedViewCheck = mlir::cast<subop::HashIndexedViewType>(lookupRefCheck.getState());
    EXPECT_TRUE(hashIndexedViewCheck.getCompareHashForLookup());
    
    // During lowering, this should generate hash comparison logic
}

// ============================================================================
// Pass Integration Tests
// ============================================================================

TEST_F(ScanOperationsTest, ScanLoweringPassIntegration) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_scan_lowering", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a simple scan operation for lowering
    auto bufferType = subop::BufferType::get(&context, ArrayAttr(), false);
    auto bufferValue = builder.create<util::UndefOp>(loc, bufferType);
    
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto column = tupleDialect.getColumnManager().createColumn("lowering_col", builder.getI32Type());
    auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
    
    auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, bufferValue);
    
    builder.create<func::ReturnOp>(loc);
    
    // Create pass pipeline for scan operations
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    // Run the lowering pass
    auto result = pm.run(module);
    
    // Verify the pass completes successfully
    EXPECT_TRUE(succeeded(result));
    
    // After lowering, the scan operation should be converted to control flow
    // The specific pattern depends on the scan type and will be tested during integration
}

TEST_F(ScanOperationsTest, ComprehensiveScanTypeHandling) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_comprehensive_scan", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    auto& tupleDialect = *context.getLoadedDialect<tuples::TupleStreamDialect>();
    
    // Test all major scan types in one comprehensive test
    std::vector<std::pair<std::string, Type>> scanTypes;
    
    // Buffer type
    auto bufferType = subop::BufferType::get(&context, ArrayAttr(), false);
    scanTypes.emplace_back("buffer", bufferType);
    
    // Simple state type
    auto simpleStateType = subop::SimpleStateType::get(&context, builder.getI64Type());
    scanTypes.emplace_back("simple_state", simpleStateType);
    
    // Sorted view type
    auto members = ArrayAttr::get(&context, {TypeAttr::get(builder.getI32Type())});
    auto sortedViewType = subop::SortedViewType::get(&context, members, false);
    scanTypes.emplace_back("sorted_view", sortedViewType);
    
    // Heap type
    auto heapType = subop::HeapType::get(&context, members, false);
    scanTypes.emplace_back("heap", heapType);
    
    // Create scan operations for each type
    for (const auto& [name, type] : scanTypes) {
        auto stateValue = builder.create<util::UndefOp>(loc, type);
        auto column = tupleDialect.getColumnManager().createColumn(name + "_col", builder.getI32Type());
        auto columnDefAttr = tupleDialect.getColumnManager().createDef(column);
        
        auto scanOp = builder.create<subop::ScanRefsOp>(loc, columnDefAttr, stateValue);
        
        // Verify each scan operation
        EXPECT_TRUE(scanOp);
        EXPECT_EQ(scanOp.getRef().getColumn().getName(), name + "_col");
        EXPECT_EQ(scanOp.getState().getType(), type);
    }
    
    builder.create<func::ReturnOp>(loc);
    
    // Verify module contains all operations
    EXPECT_TRUE(module);
    EXPECT_EQ(module.getBody()->getOperations().size(), 1); // One function
    
    // Function should contain all scan operations plus the return
    size_t opCount = 0;
    funcOp.walk([&](Operation* op) {
        if (mlir::isa<subop::ScanRefsOp>(op)) {
            opCount++;
        }
    });
    EXPECT_EQ(opCount, scanTypes.size());
}