#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Include all required dialects
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/ColumnManager.h"
#include "dialects/util/UtilDialect.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ConstRelationUnionLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load all required dialects
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOpDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        
        // Initialize column manager
        auto* tupleStreamDialect = context.getLoadedDialect<tuples::TupleStreamDialect>();
        columnManager = &tupleStreamDialect->getColumnManager();
        columnManager->setContext(&context);
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
    tuples::ColumnManager* columnManager;
    
    // Helper method to create column definitions
    tuples::ColumnDefAttr createColumn(const std::string& scope, const std::string& name, Type type) {
        auto columnDef = columnManager->createDef(scope, name);
        auto& column = columnDef.getColumn();
        column.type = type;
        return columnDef;
    }
    
    // Helper method to create constant relation with test data
    relalg::ConstRelationOp createConstRelation(ArrayRef<tuples::ColumnDefAttr> columns, ArrayRef<ArrayAttr> values) {
        return builder->create<relalg::ConstRelationOp>(
            loc,
            builder->getArrayAttr(columns),
            builder->getArrayAttr(values)
        );
    }
    
    // Helper method to create union operation
    relalg::UnionOp createUnion(Value left, Value right, relalg::SetSemantic semantic, ArrayRef<tuples::ColumnDefAttr> mapping) {
        return builder->create<relalg::UnionOp>(
            loc,
            left,
            right,
            semantic,
            builder->getArrayAttr(mapping)
        );
    }
    
    // Helper to run RelAlg → SubOp lowering pass
    LogicalResult runLoweringPass(ModuleOp module) {
        PassManager pm(&context);
        pm.addPass(relalg::createLowerRelAlgToSubOpPass());
        return pm.run(module);
    }
};

TEST_F(ConstRelationUnionLoweringTest, ConstRelationBasicLowering) {
    // Create a simple module
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create function container
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_const_relation", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create column definitions for test table (id: i32, name: text)
    auto idColumn = createColumn("test", "id", builder->getI32Type());
    auto nameColumn = createColumn("test", "name", 
        db::CharType::get(&context, false, 20)); // Non-nullable char(20)
    
    std::vector<tuples::ColumnDefAttr> columns = {idColumn, nameColumn};
    
    // Create constant data rows
    auto row1Values = builder->getArrayAttr({
        builder->getI32IntegerAttr(1),
        builder->getStringAttr("Alice")
    });
    auto row2Values = builder->getArrayAttr({
        builder->getI32IntegerAttr(2), 
        builder->getStringAttr("Bob")
    });
    
    std::vector<ArrayAttr> values = {row1Values, row2Values};
    
    // Create ConstRelationOp
    auto constOp = createConstRelation(columns, values);
    
    // Create return operation
    builder->create<relalg::ReturnOp>(loc, constOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    // Verify IR is valid before lowering
    ASSERT_TRUE(module.verify().succeeded()) << "Module verification failed before lowering";
    
    // Run lowering pass
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "ConstRelation lowering failed";
    
    // Verify IR is still valid after lowering
    ASSERT_TRUE(module.verify().succeeded()) << "Module verification failed after lowering";
    
    // Verify that ConstRelationOp was replaced with GenerateOp
    bool foundGenerateOp = false;
    module.walk([&](subop::GenerateOp generateOp) {
        foundGenerateOp = true;
        
        // Verify the generate operation has the correct structure
        EXPECT_TRUE(generateOp.getRegion().hasOneBlock());
        
        auto& block = generateOp.getRegion().front();
        
        // Count GenerateEmitOp operations (should be 2 for our test data)
        int emitCount = 0;
        block.walk([&](subop::GenerateEmitOp) { emitCount++; });
        EXPECT_EQ(emitCount, 2) << "Expected 2 GenerateEmitOp operations for 2 rows";
        
        // Verify the block ends with ReturnOp
        EXPECT_TRUE(isa<tuples::ReturnOp>(block.back()));
    });
    
    EXPECT_TRUE(foundGenerateOp) << "GenerateOp not found after lowering";
}

TEST_F(ConstRelationUnionLoweringTest, ConstRelationWithNullValues) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_const_relation_nulls", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create nullable column
    auto nullableIntType = db::NullableType::get(&context, builder->getI32Type());
    auto idColumn = createColumn("test", "id", nullableIntType);
    
    std::vector<tuples::ColumnDefAttr> columns = {idColumn};
    
    // Create rows with null and non-null values
    auto row1Values = builder->getArrayAttr({builder->getI32IntegerAttr(1)});
    auto row2Values = builder->getArrayAttr({builder->getUnitAttr()}); // null value
    
    std::vector<ArrayAttr> values = {row1Values, row2Values};
    
    auto constOp = createConstRelation(columns, values);
    builder->create<relalg::ReturnOp>(loc, constOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "ConstRelation with nulls lowering failed";
    
    // Verify null handling in generated IR
    bool foundNullOp = false;
    bool foundAsNullableOp = false;
    
    module.walk([&](db::NullOp) { foundNullOp = true; });
    module.walk([&](db::AsNullableOp) { foundAsNullableOp = true; });
    
    EXPECT_TRUE(foundNullOp) << "NullOp not found for null value handling";
    EXPECT_TRUE(foundAsNullableOp) << "AsNullableOp not found for nullable type conversion";
}

TEST_F(ConstRelationUnionLoweringTest, EmptyConstRelation) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_empty_const_relation", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create column definition but no data rows
    auto idColumn = createColumn("test", "id", builder->getI32Type());
    std::vector<tuples::ColumnDefAttr> columns = {idColumn};
    std::vector<ArrayAttr> values = {}; // empty
    
    auto constOp = createConstRelation(columns, values);
    builder->create<relalg::ReturnOp>(loc, constOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "Empty ConstRelation lowering failed";
    
    // Verify empty relation generates proper structure
    bool foundGenerateOp = false;
    module.walk([&](subop::GenerateOp generateOp) {
        foundGenerateOp = true;
        
        auto& block = generateOp.getRegion().front();
        
        // Should have no GenerateEmitOp operations
        int emitCount = 0;
        block.walk([&](subop::GenerateEmitOp) { emitCount++; });
        EXPECT_EQ(emitCount, 0) << "Expected 0 GenerateEmitOp operations for empty relation";
        
        // Should still have ReturnOp
        EXPECT_TRUE(isa<tuples::ReturnOp>(block.back()));
    });
    
    EXPECT_TRUE(foundGenerateOp) << "GenerateOp not found for empty relation";
}

TEST_F(ConstRelationUnionLoweringTest, UnionAllBasicOperation) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_union_all", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create two constant relations with same schema
    auto idColumn = createColumn("left", "id", builder->getI32Type());
    auto mappedColumn = createColumn("result", "id", builder->getI32Type());
    
    // Create left relation
    auto leftRow = builder->getArrayAttr({builder->getI32IntegerAttr(1)});
    auto leftConst = createConstRelation({idColumn}, {leftRow});
    
    // Create right relation with different column name but same type
    auto rightIdColumn = createColumn("right", "id", builder->getI32Type());
    auto rightRow = builder->getArrayAttr({builder->getI32IntegerAttr(2)});
    auto rightConst = createConstRelation({rightIdColumn}, {rightRow});
    
    // Create union operation
    auto unionOp = createUnion(
        leftConst.getResult(),
        rightConst.getResult(), 
        relalg::SetSemantic::all,
        {mappedColumn}
    );
    
    builder->create<relalg::ReturnOp>(loc, unionOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "UnionAll lowering failed";
    
    // Verify UnionAll was replaced with SubOp UnionOp
    bool foundSubOpUnion = false;
    module.walk([&](subop::UnionOp subopUnion) {
        foundSubOpUnion = true;
        
        // Verify it has two operands
        EXPECT_EQ(subopUnion.getStreams().size(), 2u) << "Union should have 2 input streams";
    });
    
    EXPECT_TRUE(foundSubOpUnion) << "SubOp UnionOp not found after UnionAll lowering";
}

TEST_F(ConstRelationUnionLoweringTest, UnionAllWithTypeConversion) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_union_all_conversion", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create relations with different nullability
    auto nonNullableType = builder->getI32Type();
    auto nullableType = db::NullableType::get(&context, builder->getI32Type());
    
    auto leftColumn = createColumn("left", "id", nonNullableType);
    auto rightColumn = createColumn("right", "id", nullableType);
    auto resultColumn = createColumn("result", "id", nullableType); // Union result should be nullable
    
    // Create left relation (non-nullable)
    auto leftRow = builder->getArrayAttr({builder->getI32IntegerAttr(1)});
    auto leftConst = createConstRelation({leftColumn}, {leftRow});
    
    // Create right relation (nullable)
    auto rightRow = builder->getArrayAttr({builder->getI32IntegerAttr(2)});
    auto rightConst = createConstRelation({rightColumn}, {rightRow});
    
    // Create union operation
    auto unionOp = createUnion(
        leftConst.getResult(),
        rightConst.getResult(),
        relalg::SetSemantic::all,
        {resultColumn}
    );
    
    builder->create<relalg::ReturnOp>(loc, unionOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "UnionAll with type conversion lowering failed";
    
    // Verify type conversion operations were inserted
    bool foundAsNullableOp = false;
    module.walk([&](db::AsNullableOp) { foundAsNullableOp = true; });
    
    EXPECT_TRUE(foundAsNullableOp) << "AsNullableOp not found for type conversion in union";
}

TEST_F(ConstRelationUnionLoweringTest, UnionDistinctBasicOperation) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_union_distinct", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create two constant relations with overlapping data
    auto idColumn = createColumn("left", "id", builder->getI32Type());
    auto mappedColumn = createColumn("result", "id", builder->getI32Type());
    
    // Create left relation with duplicate value
    auto leftRow1 = builder->getArrayAttr({builder->getI32IntegerAttr(1)});
    auto leftRow2 = builder->getArrayAttr({builder->getI32IntegerAttr(2)});
    auto leftConst = createConstRelation({idColumn}, {leftRow1, leftRow2});
    
    // Create right relation with overlapping value
    auto rightIdColumn = createColumn("right", "id", builder->getI32Type());
    auto rightRow1 = builder->getArrayAttr({builder->getI32IntegerAttr(1)}); // Duplicate
    auto rightRow2 = builder->getArrayAttr({builder->getI32IntegerAttr(3)});
    auto rightConst = createConstRelation({rightIdColumn}, {rightRow1, rightRow2});
    
    // Create union distinct operation
    auto unionOp = createUnion(
        leftConst.getResult(),
        rightConst.getResult(),
        relalg::SetSemantic::distinct,
        {mappedColumn}
    );
    
    builder->create<relalg::ReturnOp>(loc, unionOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "UnionDistinct lowering failed";
    
    // Verify distinct implementation uses hash map and deduplication
    bool foundLookupOrInsertOp = false;
    bool foundScanOp = false;
    bool foundReduceOp = false;
    
    module.walk([&](subop::LookupOrInsertOp) { foundLookupOrInsertOp = true; });
    module.walk([&](subop::ScanOp) { foundScanOp = true; });
    module.walk([&](subop::ReduceOp) { foundReduceOp = true; });
    
    EXPECT_TRUE(foundLookupOrInsertOp) << "LookupOrInsertOp not found for distinct implementation";
    EXPECT_TRUE(foundScanOp) << "ScanOp not found for distinct result scanning";
    EXPECT_TRUE(foundReduceOp) << "ReduceOp not found for distinct processing";
}

TEST_F(ConstRelationUnionLoweringTest, UnionDistinctComplexTypes) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_union_distinct_complex", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create relations with multiple columns for complex deduplication
    auto idColumn = createColumn("left", "id", builder->getI32Type());
    auto nameColumn = createColumn("left", "name", db::CharType::get(&context, false, 20));
    
    auto mappedIdColumn = createColumn("result", "id", builder->getI32Type());
    auto mappedNameColumn = createColumn("result", "name", db::CharType::get(&context, false, 20));
    
    // Create left relation
    auto leftRow = builder->getArrayAttr({
        builder->getI32IntegerAttr(1),
        builder->getStringAttr("Alice")
    });
    auto leftConst = createConstRelation({idColumn, nameColumn}, {leftRow});
    
    // Create right relation with same columns
    auto rightIdColumn = createColumn("right", "id", builder->getI32Type());
    auto rightNameColumn = createColumn("right", "name", db::CharType::get(&context, false, 20));
    auto rightRow = builder->getArrayAttr({
        builder->getI32IntegerAttr(1),
        builder->getStringAttr("Alice") // Duplicate row
    });
    auto rightConst = createConstRelation({rightIdColumn, rightNameColumn}, {rightRow});
    
    // Create union distinct operation with multiple columns
    auto unionOp = createUnion(
        leftConst.getResult(),
        rightConst.getResult(),
        relalg::SetSemantic::distinct,
        {mappedIdColumn, mappedNameColumn}
    );
    
    builder->create<relalg::ReturnOp>(loc, unionOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "UnionDistinct with complex types lowering failed";
    
    // Verify that deduplication works with multiple key columns
    bool foundMapType = false;
    module.walk([&](subop::GenericCreateOp createOp) {
        if (auto mapType = dyn_cast<subop::MapType>(createOp.getType())) {
            foundMapType = true;
            
            // Verify the map has the correct key structure for both columns
            auto keyMembers = mapType.getKeyMembers();
            EXPECT_EQ(keyMembers.getNames().size(), 2u) << "Expected 2 key members for deduplication";
        }
    });
    
    EXPECT_TRUE(foundMapType) << "MapType not found for complex deduplication";
}

TEST_F(ConstRelationUnionLoweringTest, UnionAllFailsForDistinctInput) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_union_pattern_matching", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    auto idColumn = createColumn("test", "id", builder->getI32Type());
    auto mappedColumn = createColumn("result", "id", builder->getI32Type());
    
    auto leftRow = builder->getArrayAttr({builder->getI32IntegerAttr(1)});
    auto leftConst = createConstRelation({idColumn}, {leftRow});
    
    auto rightRow = builder->getArrayAttr({builder->getI32IntegerAttr(2)});
    auto rightConst = createConstRelation({idColumn}, {rightRow});
    
    // Create union distinct operation (should NOT match UnionAllLowering pattern)
    auto unionOp = createUnion(
        leftConst.getResult(),
        rightConst.getResult(),
        relalg::SetSemantic::distinct, // This should make UnionAllLowering fail
        {mappedColumn}
    );
    
    builder->create<relalg::ReturnOp>(loc, unionOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "Union lowering failed";
    
    // Verify that UnionAll pattern did not match (should use distinct implementation)
    bool foundSimpleUnionOp = false;
    bool foundLookupOrInsertOp = false;
    
    module.walk([&](subop::UnionOp) { foundSimpleUnionOp = true; });
    module.walk([&](subop::LookupOrInsertOp) { foundLookupOrInsertOp = true; });
    
    // For distinct semantic, should use lookup-based implementation, not simple union
    EXPECT_FALSE(foundSimpleUnionOp) << "Simple UnionOp should not be used for distinct semantic";
    EXPECT_TRUE(foundLookupOrInsertOp) << "LookupOrInsertOp should be used for distinct semantic";
}

TEST_F(ConstRelationUnionLoweringTest, LargeConstRelationPerformance) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_large_const_relation", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    auto idColumn = createColumn("test", "id", builder->getI32Type());
    std::vector<tuples::ColumnDefAttr> columns = {idColumn};
    
    // Create a larger constant relation (100 rows)
    std::vector<ArrayAttr> values;
    for (int i = 0; i < 100; ++i) {
        auto rowValues = builder->getArrayAttr({builder->getI32IntegerAttr(i)});
        values.push_back(rowValues);
    }
    
    auto constOp = createConstRelation(columns, values);
    builder->create<relalg::ReturnOp>(loc, constOp.getResult());
    builder->create<func::ReturnOp>(loc);
    
    ASSERT_TRUE(runLoweringPass(module).succeeded()) << "Large ConstRelation lowering failed";
    
    // Verify all 100 rows are properly generated
    bool foundGenerateOp = false;
    module.walk([&](subop::GenerateOp generateOp) {
        foundGenerateOp = true;
        
        auto& block = generateOp.getRegion().front();
        
        // Count GenerateEmitOp operations (should be 100)
        int emitCount = 0;
        block.walk([&](subop::GenerateEmitOp) { emitCount++; });
        EXPECT_EQ(emitCount, 100) << "Expected 100 GenerateEmitOp operations for 100 rows";
    });
    
    EXPECT_TRUE(foundGenerateOp) << "GenerateOp not found for large relation";
}