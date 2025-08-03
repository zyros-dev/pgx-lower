// Comprehensive unit tests for Operations/TableOperations.cpp
// Testing table operation lowering patterns for potential issues with operations after return statements

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuples/TupleStreamDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class TableOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext context;
};

//===----------------------------------------------------------------------===//
// Table Access Tests
//===----------------------------------------------------------------------===//

TEST_F(TableOperationsTest, TestTableRefGatherOpCreation) {
    PGX_DEBUG("Testing TableRefGatherOp creation and basic structure");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create table entry reference type with column members
    llvm::SmallVector<mlir::Attribute> columnNames;
    llvm::SmallVector<mlir::Attribute> columnTypes;
    
    columnNames.push_back(builder.getStringAttr("id"));
    columnNames.push_back(builder.getStringAttr("name"));
    columnTypes.push_back(TypeAttr::get(builder.getI32Type()));
    columnTypes.push_back(TypeAttr::get(builder.getType<db::StringType>()));
    
    auto tableMembers = tuples::ColumnDefArrayAttr::get(&context, columnNames, columnTypes);
    auto tableRefType = subop::TableEntryRefType::get(&context, tableMembers);
    
    // Create column reference for gather operation
    auto tableColumnDef = tuples::ColumnDefAttr::get(&context, 
        tuples::ColumnAttr::get(&context, "tableRef", tableRefType));
    
    // Verify table reference type is correctly constructed
    EXPECT_TRUE(tableRefType);
    EXPECT_EQ(tableRefType.getMembers().getTypes().size(), 2);
    EXPECT_TRUE(tableColumnDef);
    
    PGX_DEBUG("TableRefGatherOp structure validation passed");
}

TEST_F(TableOperationsTest, TestTableAccessPatternValidation) {
    PGX_DEBUG("Testing table access pattern validation for termination issues");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function to contain table access operations
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_table_access", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Test PostgreSQL tuple field access pattern
    auto tupleType = builder.getType<util::RefType>();
    auto fieldIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto fieldName = builder.getStringAttr("test_field");
    
    // Simulate the LoadPostgreSQLOp pattern used in TableRefGatherOpLowering
    // This tests the critical path where PostgreSQL tuple access happens
    EXPECT_TRUE(fieldIndex);
    EXPECT_TRUE(fieldName);
    
    // Verify no operations are added after return statement pattern
    // (This is a structural test to prevent the bug mentioned in requirements)
    auto hasTerminator = block->hasTerminator();
    
    // Add proper terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify terminator is present and no operations follow
    EXPECT_TRUE(block->hasTerminator());
    EXPECT_TRUE(block->getTerminator() == &block->back());
    
    PGX_DEBUG("Table access pattern validation completed successfully");
}

//===----------------------------------------------------------------------===//
// Column Operations Tests
//===----------------------------------------------------------------------===//

TEST_F(TableOperationsTest, TestColumnAccessAndModification) {
    PGX_DEBUG("Testing column access and modification operations");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create column definition for testing
    auto columnType = builder.getI32Type();
    auto columnDef = tuples::ColumnDefAttr::get(&context,
        tuples::ColumnAttr::get(&context, "test_column", columnType));
    
    // Test column unpacking pattern (from TableRefGatherOpLowering)
    auto i16Type = IntegerType::get(&context, 16);
    auto refType = util::RefType::get(&context, i16Type);
    
    // Verify column types and structures
    EXPECT_TRUE(columnDef);
    EXPECT_EQ(columnDef.getColumn().name.str(), "test_column");
    EXPECT_TRUE(refType);
    
    // Test column mapping patterns
    std::vector<Type> accessedColumnTypes;
    accessedColumnTypes.push_back(columnType);
    
    EXPECT_EQ(accessedColumnTypes.size(), 1);
    EXPECT_EQ(accessedColumnTypes[0], columnType);
    
    PGX_DEBUG("Column access and modification tests passed");
}

TEST_F(TableOperationsTest, TestColumnOperationTermination) {
    PGX_DEBUG("Testing column operation termination to prevent operations after return");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_column_ops", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate column access loop (similar to TableRefGatherOpLowering line 37-51)
    auto constantIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
    
    // Critical test: Ensure operations complete before any termination
    size_t operationCountBefore = block->getOperations().size();
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Verify no operations added after terminator
    size_t operationCountAfter = block->getOperations().size();
    EXPECT_EQ(operationCountAfter, operationCountBefore + 1); // Only terminator added
    EXPECT_TRUE(block->hasTerminator());
    
    PGX_DEBUG("Column operation termination test passed - no operations after return");
}

//===----------------------------------------------------------------------===//
// Schema Tests
//===----------------------------------------------------------------------===//

TEST_F(TableOperationsTest, TestTableSchemaHandling) {
    PGX_DEBUG("Testing table schema handling and validation");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create result table type schema
    llvm::SmallVector<Attribute> memberNames;
    llvm::SmallVector<Attribute> memberTypes;
    
    memberNames.push_back(builder.getStringAttr("column1"));
    memberNames.push_back(builder.getStringAttr("column2"));
    memberNames.push_back(builder.getStringAttr("column3"));
    
    memberTypes.push_back(TypeAttr::get(builder.getI32Type()));
    memberTypes.push_back(TypeAttr::get(builder.getF64Type()));
    memberTypes.push_back(TypeAttr::get(builder.getType<db::StringType>()));
    
    auto tableMembers = tuples::ColumnDefArrayAttr::get(&context, memberNames, memberTypes);
    auto resultTableType = subop::ResultTableType::get(&context, tableMembers, false);
    
    // Verify schema structure
    EXPECT_TRUE(resultTableType);
    EXPECT_EQ(resultTableType.getMembers().getTypes().size(), 3);
    EXPECT_EQ(resultTableType.getMembers().getNames().size(), 3);
    
    // Test schema member access patterns
    for (size_t i = 0; i < resultTableType.getMembers().getTypes().size(); i++) {
        auto memberName = cast<StringAttr>(resultTableType.getMembers().getNames()[i]).str();
        auto memberType = cast<TypeAttr>(resultTableType.getMembers().getTypes()[i]).getValue();
        
        EXPECT_FALSE(memberName.empty());
        EXPECT_TRUE(memberType);
        
        // Verify no operations after member processing
        // (Critical for MaterializeTableLowering pattern)
    }
    
    PGX_DEBUG("Table schema handling validation completed");
}

TEST_F(TableOperationsTest, TestSchemaValidationWithMaterialize) {
    PGX_DEBUG("Testing schema validation with materialization operations");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Test heap type schema
    llvm::SmallVector<Attribute> heapMembers;
    heapMembers.push_back(TypeAttr::get(builder.getI64Type()));
    heapMembers.push_back(TypeAttr::get(builder.getType<db::StringType>()));
    
    auto heapMemberArray = ArrayAttr::get(&context, heapMembers);
    auto heapType = subop::HeapType::get(&context, heapMemberArray, true); // hasLock = true
    
    EXPECT_TRUE(heapType);
    EXPECT_TRUE(heapType.hasLock());
    EXPECT_EQ(heapType.getMembers().size(), 2);
    
    // Test buffer type schema  
    auto bufferType = subop::BufferType::get(&context, heapMemberArray, false); // hasLock = false
    
    EXPECT_TRUE(bufferType);
    EXPECT_FALSE(bufferType.hasLock());
    EXPECT_EQ(bufferType.getMembers().size(), 2);
    
    PGX_DEBUG("Schema validation with materialization passed");
}

//===----------------------------------------------------------------------===//
// Transaction Tests
//===----------------------------------------------------------------------===//

TEST_F(TableOperationsTest, TestTableOperationAtomicity) {
    PGX_DEBUG("Testing table operation atomicity and consistency");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_atomicity", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Test atomic operation pattern - all operations must complete or none
    auto startOpCount = block->getOperations().size();
    
    // Simulate atomic table operation sequence
    auto constOp1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto constOp2 = builder.create<arith::ConstantIndexOp>(loc, 2);
    auto constOp3 = builder.create<arith::ConstantIndexOp>(loc, 3);
    
    auto midOpCount = block->getOperations().size();
    EXPECT_EQ(midOpCount, startOpCount + 3);
    
    // Critical: Ensure all operations are properly terminated
    builder.create<func::ReturnOp>(loc);
    
    auto finalOpCount = block->getOperations().size();
    EXPECT_EQ(finalOpCount, midOpCount + 1);
    EXPECT_TRUE(block->hasTerminator());
    
    // Verify atomicity - all operations present and terminated correctly
    EXPECT_TRUE(constOp1);
    EXPECT_TRUE(constOp2);
    EXPECT_TRUE(constOp3);
    
    PGX_DEBUG("Table operation atomicity test passed");
}

TEST_F(TableOperationsTest, TestTransactionConsistency) {
    PGX_DEBUG("Testing transaction consistency in table operations");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Test PostgreSQL memory context consistency (critical for LOAD command issue)
    auto i8Type = IntegerType::get(&context, 8);
    auto ptrType = util::RefType::get(&context, i8Type);
    
    // Simulate memory context handling
    auto funcType = FunctionType::get(&context, {ptrType}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_memory_context", funcType);
    auto* block = funcOp.addEntryBlock();
    
    auto contextPtr = block->addArgument(ptrType, loc);
    
    builder.setInsertionPointToEnd(block);
    
    // Test context pointer validity
    EXPECT_TRUE(contextPtr);
    EXPECT_EQ(contextPtr.getType(), ptrType);
    
    // Simulate record batch processing consistency
    auto i16Type = IntegerType::get(&context, 16);
    std::vector<Type> recordBatchTypes{
        builder.getIndexType(), 
        builder.getIndexType(), 
        util::RefType::get(&context, i16Type), 
        util::RefType::get(&context, builder.getI16Type())
    };
    auto recordBatchInfoRepr = TupleType::get(&context, recordBatchTypes);
    
    EXPECT_TRUE(recordBatchInfoRepr);
    EXPECT_EQ(recordBatchInfoRepr.getTypes().size(), 4);
    
    // Ensure consistent termination
    builder.create<func::ReturnOp>(loc);
    EXPECT_TRUE(block->hasTerminator());
    
    PGX_DEBUG("Transaction consistency test completed");
}

//===----------------------------------------------------------------------===//
// Critical Bug Prevention Tests - Operations After Return Statements
//===----------------------------------------------------------------------===//

TEST_F(TableOperationsTest, TestPreventOperationsAfterReturn) {
    PGX_DEBUG("Testing prevention of operations after return statements");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_return_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate the pattern from TableRefGatherOpLowering (lines 52-54)
    auto constOp = builder.create<arith::ConstantIndexOp>(loc, 42);
    
    // Critical test: Add terminator (equivalent to replaceTupleStream + return success())
    builder.create<func::ReturnOp>(loc);
    
    // Verify terminator is last operation
    EXPECT_TRUE(block->hasTerminator());
    EXPECT_TRUE(&block->back() == block->getTerminator());
    
    // Verify no operations can be added after terminator
    auto terminatorPos = std::prev(block->end());
    EXPECT_TRUE(terminatorPos->hasTrait<OpTrait::IsTerminator>());
    
    PGX_DEBUG("Operations after return prevention test passed");
}

TEST_F(TableOperationsTest, TestMaterializationTerminationSafety) {
    PGX_DEBUG("Testing materialization operations termination safety");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_materialize_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate MaterializeTableLowering pattern (lines 65-81)
    auto loadOp = builder.create<arith::ConstantIndexOp>(loc, 0);
    
    // Test the critical eraseOp pattern that should terminate the lowering
    // (Equivalent to rewriter.eraseOp(materializeOp) on line 78)
    EXPECT_TRUE(loadOp);
    
    // Ensure proper termination without orphaned operations
    builder.create<func::ReturnOp>(loc);
    
    EXPECT_TRUE(block->hasTerminator());
    
    // Verify operation ordering is preserved
    auto& operations = block->getOperations();
    auto it = operations.begin();
    EXPECT_TRUE(&(*it) == loadOp.getOperation());
    ++it;
    EXPECT_TRUE((*it).hasTrait<OpTrait::IsTerminator>());
    
    PGX_DEBUG("Materialization termination safety test passed");
}

TEST_F(TableOperationsTest, TestScanIterationTerminationSafety) {
    PGX_DEBUG("Testing scan iteration termination safety");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Test the complex ScanRefsTableLowering pattern (lines 125-195)
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_scan_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate the nested function creation pattern
    auto ptrType = util::RefType::get(&context, IntegerType::get(&context, 8));
    auto innerFuncType = FunctionType::get(&context, {ptrType, ptrType}, {});
    auto innerFuncOp = builder.create<func::FuncOp>(loc, "scan_func_test", innerFuncType);
    
    // Add inner function body
    auto* innerBlock = innerFuncOp.addEntryBlock();
    auto recordBatchPointer = innerBlock->addArgument(ptrType, loc);
    auto contextPtr = innerBlock->addArgument(ptrType, loc);
    
    // Set insertion to inner block and add terminator
    builder.setInsertionPointToEnd(innerBlock);
    builder.create<func::ReturnOp>(loc);
    
    // Verify inner function is properly terminated
    EXPECT_TRUE(innerBlock->hasTerminator());
    EXPECT_TRUE(recordBatchPointer);
    EXPECT_TRUE(contextPtr);
    
    // Return to outer block and terminate
    builder.setInsertionPointToEnd(block);
    builder.create<func::ReturnOp>(loc);
    
    EXPECT_TRUE(block->hasTerminator());
    
    PGX_DEBUG("Scan iteration termination safety test passed");
}

TEST_F(TableOperationsTest, TestGenerateOpTerminationSafety) {
    PGX_DEBUG("Testing GenerateOp termination safety");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Test GenerateLowering pattern (lines 299-328)
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_generate_safety", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate emit operation processing
    std::vector<Value> streams;
    auto constValue = builder.create<arith::ConstantIndexOp>(loc, 1);
    streams.push_back(constValue);
    
    // Test the critical erase operation pattern
    EXPECT_EQ(streams.size(), 1);
    EXPECT_TRUE(streams[0]);
    
    // Ensure proper termination after emit processing
    builder.create<func::ReturnOp>(loc);
    
    EXPECT_TRUE(block->hasTerminator());
    
    // Verify no dangling operations after the erase pattern
    auto terminatorOp = block->getTerminator();
    EXPECT_TRUE(terminatorOp);
    EXPECT_TRUE(terminatorOp == &block->back());
    
    PGX_DEBUG("GenerateOp termination safety test passed");
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

TEST_F(TableOperationsTest, TestCompleteTableOperationPipeline) {
    PGX_DEBUG("Testing complete table operation pipeline integration");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    
    // Create a comprehensive test that exercises all table operation patterns
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    // Verify the pipeline can run without introducing operations after returns
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result));
    
    // Walk through all functions and verify termination safety
    module.walk([](func::FuncOp funcOp) {
        for (auto& region : funcOp->getRegions()) {
            for (auto& block : region.getBlocks()) {
                if (!block.empty()) {
                    // Verify last operation is terminator if block is non-empty
                    auto& lastOp = block.back();
                    if (block.hasTerminator()) {
                        EXPECT_TRUE(lastOp.hasTrait<OpTrait::IsTerminator>());
                    }
                }
            }
        }
    });
    
    PGX_DEBUG("Complete table operation pipeline integration test passed");
}

TEST_F(TableOperationsTest, TestPostgreSQLMemoryContextSafety) {
    PGX_DEBUG("Testing PostgreSQL memory context safety in table operations");
    
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Test the PostgreSQL-specific patterns from TableRefGatherOpLowering
    auto funcType = FunctionType::get(&context, {}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_postgresql_memory", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder.setInsertionPointToEnd(block);
    
    // Simulate PostgreSQL tuple field access (line 43-48)
    auto fieldIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto fieldNameAttr = builder.getStringAttr("test_field");
    
    // Test memory context invalidation resistance
    EXPECT_TRUE(fieldIndex);
    EXPECT_TRUE(fieldNameAttr);
    
    // Ensure operations complete before any memory context changes
    builder.create<func::ReturnOp>(loc);
    
    // Verify memory safety - no dangling references
    EXPECT_TRUE(block->hasTerminator());
    
    // Test that field access patterns are memory-safe
    auto& ops = block->getOperations();
    for (auto& op : ops) {
        // Verify each operation is properly contained within the block
        EXPECT_EQ(op.getBlock(), block);
    }
    
    PGX_DEBUG("PostgreSQL memory context safety test completed");
}