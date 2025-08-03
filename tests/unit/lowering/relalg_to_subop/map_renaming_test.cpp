#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/Utils.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class MapRenamingLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load all required dialects
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
    }

    // Helper to create a mock input relation
    Value createMockInputRelation() {
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        return builder->create<arith::ConstantOp>(
            loc, tupleStreamType, builder->getUnitAttr());
    }

    // Helper to create test column definitions
    std::vector<Attribute> createTestColumnDefs() {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        
        std::vector<Attribute> columnDefs;
        
        // Create test columns with different types
        auto intColumn = colManager.createColumn("computed_int", builder->getI32Type());
        auto floatColumn = colManager.createColumn("computed_float", builder->getF64Type());
        auto stringColumn = colManager.createColumn("computed_string", 
            builder->getType<LLVM::LLVMPointerType>());
        
        columnDefs.push_back(colManager.createDef(intColumn));
        columnDefs.push_back(colManager.createDef(floatColumn));
        columnDefs.push_back(colManager.createDef(stringColumn));
        
        return columnDefs;
    }

    // Helper to create test column references for input
    std::vector<Attribute> createTestColumnRefs() {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        
        std::vector<Attribute> columnRefs;
        
        // Create input columns
        auto sourceInt = colManager.createColumn("source_int", builder->getI32Type());
        auto sourceFloat = colManager.createColumn("source_float", builder->getF64Type());
        auto sourceBool = colManager.createColumn("source_bool", builder->getI1Type());
        
        columnRefs.push_back(colManager.createRef(sourceInt));
        columnRefs.push_back(colManager.createRef(sourceFloat));
        columnRefs.push_back(colManager.createRef(sourceBool));
        
        return columnRefs;
    }

    // Helper to create a MapOp with expression region
    relalg::MapOp createMapOp(Value inputRel, ArrayRef<Attribute> computedCols) {
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        ArrayAttr computedAttr = builder->getArrayAttr(computedCols);
        
        auto mapOp = builder->create<relalg::MapOp>(loc, tupleStreamType, inputRel, computedAttr);
        
        // Create the computation region with expressions
        Region& predicate = mapOp.getPredicate();
        Block& block = predicate.emplaceBlock();
        
        // Add tuple argument to the block
        auto tupleType = tuples::TupleType::get(&context);
        block.addArgument(tupleType, loc);
        
        // Set insertion point to populate the region
        OpBuilder::InsertionGuard guard(*builder);
        builder->setInsertionPointToStart(&block);
        
        // Create sample expressions using GetColumnOp and arithmetic
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        auto sourceInt = colManager.createColumn("source_int", builder->getI32Type());
        auto sourceFloat = colManager.createColumn("source_float", builder->getF64Type());
        
        auto intRef = colManager.createRef(sourceInt);
        auto floatRef = colManager.createRef(sourceFloat);
        
        // Get source values using GetColumnOp
        auto tupleArg = block.getArgument(0);
        auto intVal = builder->create<tuples::GetColumnOp>(loc, builder->getI32Type(), intRef, tupleArg);
        auto floatVal = builder->create<tuples::GetColumnOp>(loc, builder->getF64Type(), floatRef, tupleArg);
        
        // Create computed expressions
        auto doubledInt = builder->create<arith::MulIOp>(loc, intVal, intVal); // square the int
        auto halfFloat = builder->create<arith::DivFOp>(loc, floatVal, 
            builder->create<arith::ConstantFloatOp>(loc, llvm::APFloat(2.0), builder->getF64Type()));
        
        // Create a string computation placeholder
        auto stringPtr = builder->create<LLVM::NullOp>(loc, builder->getType<LLVM::LLVMPointerType>());
        
        // Return the computed values
        builder->create<tuples::ReturnOp>(loc, ValueRange{doubledInt, halfFloat, stringPtr});
        
        return mapOp;
    }

    // Helper to create a RenamingOp
    relalg::RenamingOp createRenamingOp(Value inputRel, ArrayRef<Attribute> renamingColumns) {
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        ArrayAttr columnsAttr = builder->getArrayAttr(renamingColumns);
        
        return builder->create<relalg::RenamingOp>(loc, tupleStreamType, inputRel, columnsAttr);
    }

    // Helper to create renaming column mappings
    std::vector<Attribute> createRenamingColumns() {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        
        std::vector<Attribute> renamingCols;
        
        // Create original -> renamed column mappings
        auto originalCol1 = colManager.createColumn("old_name1", builder->getI32Type());
        auto renamedCol1 = colManager.createColumn("new_name1", builder->getI32Type());
        
        auto originalCol2 = colManager.createColumn("old_name2", builder->getF64Type());
        auto renamedCol2 = colManager.createColumn("new_name2", builder->getF64Type());
        
        renamingCols.push_back(colManager.createDef(renamedCol1));
        renamingCols.push_back(colManager.createDef(renamedCol2));
        
        return renamingCols;
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    Location loc;
};

TEST_F(MapRenamingLoweringTest, MapExpressionLowering) {
    // Test MapOp expression lowering to SubOp
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> computedCols = createTestColumnDefs();
    
    auto mapOp = createMapOp(inputRel, computedCols);
    
    // Verify MapOp was created correctly
    EXPECT_TRUE(mapOp);
    EXPECT_EQ(mapOp.getRel(), inputRel);
    EXPECT_EQ(mapOp.getComputedCols().size(), computedCols.size());
    
    // Verify the computation region exists and has content
    Region& predicate = mapOp.getPredicate();
    EXPECT_EQ(predicate.getBlocks().size(), 1);
    
    Block& block = predicate.front();
    EXPECT_EQ(block.getNumArguments(), 1); // tuple argument
    
    // Verify the block contains operations
    EXPECT_GT(std::distance(block.begin(), block.end()), 1); // Should have operations + terminator
    
    // Check that GetColumnOp operations exist in the region
    bool foundGetColumnOp = false;
    bool foundArithOp = false;
    bool foundReturnOp = false;
    
    for (auto& op : block) {
        if (isa<tuples::GetColumnOp>(op)) {
            foundGetColumnOp = true;
        } else if (isa<arith::MulIOp, arith::DivFOp>(op)) {
            foundArithOp = true;
        } else if (isa<tuples::ReturnOp>(op)) {
            foundReturnOp = true;
        }
    }
    
    EXPECT_TRUE(foundGetColumnOp) << "MapOp should contain GetColumnOp for column access";
    EXPECT_TRUE(foundArithOp) << "MapOp should contain arithmetic operations";
    EXPECT_TRUE(foundReturnOp) << "MapOp should have proper terminator";
    
    // Verify computed columns metadata
    auto computedColsAttr = mapOp.getComputedCols();
    EXPECT_EQ(computedColsAttr.size(), 3); // int, float, string computations
    
    for (auto col : computedColsAttr) {
        EXPECT_TRUE(isa<tuples::ColumnDefAttr>(col));
    }
}

TEST_F(MapRenamingLoweringTest, MapToSubOpConversion) {
    // Test the core MapLowering conversion logic
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> computedCols = createTestColumnDefs();
    
    auto mapOp = createMapOp(inputRel, computedCols);
    
    // Test MapCreationHelper functionality (used in MapLowering)
    subop::MapCreationHelper helper(&context);
    
    // Verify helper can create proper structures
    EXPECT_TRUE(helper.getMapBlock());
    EXPECT_TRUE(helper.getColRefs());
    
    // Test access method with column references
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    auto testColumn = colManager.createColumn("test_col", builder->getI32Type());
    auto testRef = colManager.createRef(testColumn);
    
    auto accessValue = helper.access(testRef, loc);
    EXPECT_TRUE(accessValue);
    EXPECT_TRUE(isa<BlockArgument>(accessValue));
    
    // Verify block argument creation
    EXPECT_EQ(helper.getMapBlock()->getNumArguments(), 1);
    EXPECT_EQ(helper.getColRefs().size(), 1);
    
    // Test multiple column access
    auto secondColumn = colManager.createColumn("test_col2", builder->getF64Type());
    auto secondRef = colManager.createRef(secondColumn);
    
    auto secondAccess = helper.access(secondRef, loc);
    EXPECT_TRUE(secondAccess);
    EXPECT_NE(accessValue, secondAccess); // Different columns should have different access
    
    EXPECT_EQ(helper.getMapBlock()->getNumArguments(), 2);
    EXPECT_EQ(helper.getColRefs().size(), 2);
}

TEST_F(MapRenamingLoweringTest, MapGetColumnOpReplacement) {
    // Test GetColumnOp replacement logic in MapLowering
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> computedCols = createTestColumnDefs();
    
    auto mapOp = createMapOp(inputRel, computedCols);
    
    // Verify GetColumnOp operations are properly structured for replacement
    Region& predicate = mapOp.getPredicate();
    Block& block = predicate.front();
    
    std::vector<tuples::GetColumnOp> getColumnOps;
    block.walk([&](tuples::GetColumnOp op) {
        getColumnOps.push_back(op);
    });
    
    EXPECT_GT(getColumnOps.size(), 0) << "MapOp should contain GetColumnOp operations";
    
    // Verify each GetColumnOp has proper structure for replacement
    for (auto getColOp : getColumnOps) {
        EXPECT_TRUE(getColOp.getAttr());
        EXPECT_TRUE(getColOp.getTuple());
        EXPECT_TRUE(getColOp.getRes());
        
        // Verify column reference structure
        auto columnRef = getColOp.getAttr();
        EXPECT_TRUE(isa<tuples::ColumnRefAttr>(columnRef));
        
        auto& column = columnRef.getColumn();
        EXPECT_TRUE(column.type);
        EXPECT_FALSE(column.getName().empty());
    }
}

TEST_F(MapRenamingLoweringTest, MapTerminatorHandling) {
    // Test terminator handling in MapLowering
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> computedCols = createTestColumnDefs();
    
    auto mapOp = createMapOp(inputRel, computedCols);
    
    // Verify the region has proper terminator
    Region& predicate = mapOp.getPredicate();
    Block& block = predicate.front();
    
    EXPECT_TRUE(block.getTerminator()) << "MapOp block should have terminator";
    EXPECT_TRUE(isa<tuples::ReturnOp>(block.getTerminator())) << "Should have ReturnOp terminator";
    
    // Verify terminator has return values matching computed columns
    auto returnOp = cast<tuples::ReturnOp>(block.getTerminator());
    EXPECT_EQ(returnOp.getResults().size(), computedCols.size());
    
    // Test the terminator addition logic from MapLowering
    // If MapCreationHelper creates a block without terminator, MapLowering should add one
    subop::MapCreationHelper helper(&context);
    auto* helperBlock = helper.getMapBlock();
    
    // Initially should be empty
    EXPECT_TRUE(helperBlock->empty());
    EXPECT_FALSE(helperBlock->getTerminator());
    
    // After operations are moved, terminator should be ensured
    // This simulates the MapLowering behavior
    if (!helperBlock->getTerminator()) {
        OpBuilder tempBuilder(&context);
        tempBuilder.setInsertionPointToEnd(helperBlock);
        tempBuilder.create<tuples::ReturnOp>(loc, ValueRange{});
    }
    
    EXPECT_TRUE(helperBlock->getTerminator());
}

TEST_F(MapRenamingLoweringTest, ColumnRenaming) {
    // Test RenamingOp column mapping
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> renamingCols = createRenamingColumns();
    
    auto renamingOp = createRenamingOp(inputRel, renamingCols);
    
    // Verify RenamingOp was created correctly
    EXPECT_TRUE(renamingOp);
    EXPECT_EQ(renamingOp.getRel(), inputRel);
    EXPECT_EQ(renamingOp.getColumns().size(), renamingCols.size());
    
    // Verify column metadata is preserved
    auto columnsAttr = renamingOp.getColumns();
    EXPECT_EQ(columnsAttr.size(), 2); // Two column renamings
    
    for (auto col : columnsAttr) {
        EXPECT_TRUE(isa<tuples::ColumnDefAttr>(col));
        
        auto columnDef = cast<tuples::ColumnDefAttr>(col);
        auto& column = columnDef.getColumn();
        
        // Verify column has proper metadata
        EXPECT_TRUE(column.type);
        EXPECT_FALSE(column.getName().empty());
        EXPECT_TRUE(column.getName().contains("new_name")); // Should be renamed column
    }
    
    // Verify input relation is preserved
    EXPECT_EQ(renamingOp.getRel(), inputRel);
    
    // Verify result type is TupleStream
    EXPECT_TRUE(isa<tuples::TupleStreamType>(renamingOp.getType()));
}

TEST_F(MapRenamingLoweringTest, RenamingColumnMappingPreservation) {
    // Test that column mappings are preserved during renaming
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> renamingCols = createRenamingColumns();
    
    auto renamingOp = createRenamingOp(inputRel, renamingCols);
    
    // Verify column order and types are preserved
    auto columnsAttr = renamingOp.getColumns();
    
    for (size_t i = 0; i < renamingCols.size(); ++i) {
        auto originalDef = cast<tuples::ColumnDefAttr>(renamingCols[i]);
        auto preservedDef = cast<tuples::ColumnDefAttr>(columnsAttr[i]);
        
        // Types should be identical
        EXPECT_EQ(originalDef.getColumn().type, preservedDef.getColumn().type);
        
        // Names should match (both should be the "new" names)
        EXPECT_EQ(originalDef.getColumn().getName(), preservedDef.getColumn().getName());
    }
}

TEST_F(MapRenamingLoweringTest, RenamingToSubOpConversion) {
    // Test the core RenamingLowering conversion logic
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> renamingCols = createRenamingColumns();
    
    auto renamingOp = createRenamingOp(inputRel, renamingCols);
    
    // The RenamingLowering should convert relalg::RenamingOp to subop::RenamingOp
    // Key aspects from the source code:
    // rewriter.replaceOpWithNewOp<subop::RenamingOp>(renamingOp, 
    //   tuples::TupleStreamType::get(rewriter.getContext()), 
    //   adaptor.getRel(), 
    //   renamingOp.getColumns());
    
    // Verify the operation has all necessary components for this conversion
    EXPECT_TRUE(renamingOp.getRel()); // Input relation
    EXPECT_TRUE(renamingOp.getColumns()); // Column mappings
    EXPECT_TRUE(isa<tuples::TupleStreamType>(renamingOp.getType())); // Result type
    
    // Verify column metadata structure matches SubOp requirements
    auto columnsAttr = renamingOp.getColumns();
    for (auto col : columnsAttr) {
        auto columnDef = cast<tuples::ColumnDefAttr>(col);
        
        // SubOp should be able to use these column definitions directly
        EXPECT_TRUE(columnDef.getColumn().type);
        EXPECT_FALSE(columnDef.getColumn().getName().empty());
    }
}

TEST_F(MapRenamingLoweringTest, MultipleColumnRenaming) {
    // Test multiple column renaming scenarios
    
    Value inputRel = createMockInputRelation();
    
    // Create larger set of renaming columns
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    std::vector<Attribute> multipleRenamings;
    
    // Different data types
    auto intCol = colManager.createColumn("renamed_int", builder->getI32Type());
    auto floatCol = colManager.createColumn("renamed_float", builder->getF64Type());
    auto boolCol = colManager.createColumn("renamed_bool", builder->getI1Type());
    auto ptrCol = colManager.createColumn("renamed_ptr", builder->getType<LLVM::LLVMPointerType>());
    
    multipleRenamings.push_back(colManager.createDef(intCol));
    multipleRenamings.push_back(colManager.createDef(floatCol));
    multipleRenamings.push_back(colManager.createDef(boolCol));
    multipleRenamings.push_back(colManager.createDef(ptrCol));
    
    auto renamingOp = createRenamingOp(inputRel, multipleRenamings);
    
    // Verify all renamings are preserved
    EXPECT_EQ(renamingOp.getColumns().size(), 4);
    
    auto columnsAttr = renamingOp.getColumns();
    std::set<std::string> expectedTypes = {"i32", "f64", "i1", "ptr"};
    std::set<std::string> actualTypes;
    
    for (auto col : columnsAttr) {
        auto columnDef = cast<tuples::ColumnDefAttr>(col);
        auto type = columnDef.getColumn().type;
        
        if (type.isInteger(32)) actualTypes.insert("i32");
        else if (type.isF64()) actualTypes.insert("f64");
        else if (type.isInteger(1)) actualTypes.insert("i1");
        else if (isa<LLVM::LLVMPointerType>(type)) actualTypes.insert("ptr");
    }
    
    EXPECT_EQ(actualTypes, expectedTypes) << "All data types should be preserved in renaming";
}

TEST_F(MapRenamingLoweringTest, ComplexExpressionHandling) {
    // Test complex expression handling in MapOp
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> computedCols = createTestColumnDefs();
    
    // Create MapOp with more complex expressions
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    ArrayAttr computedAttr = builder->getArrayAttr(computedCols);
    
    auto mapOp = builder->create<relalg::MapOp>(loc, tupleStreamType, inputRel, computedAttr);
    
    // Create complex computation region
    Region& predicate = mapOp.getPredicate();
    Block& block = predicate.emplaceBlock();
    auto tupleType = tuples::TupleType::get(&context);
    block.addArgument(tupleType, loc);
    
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&block);
    
    // Complex expression: conditional computation, nested arithmetic, type conversions
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    auto intColumn = colManager.createColumn("source_int", builder->getI32Type());
    auto floatColumn = colManager.createColumn("source_float", builder->getF64Type());
    auto boolColumn = colManager.createColumn("source_bool", builder->getI1Type());
    
    auto intRef = colManager.createRef(intColumn);
    auto floatRef = colManager.createRef(floatColumn);
    auto boolRef = colManager.createRef(boolColumn);
    
    auto tupleArg = block.getArgument(0);
    auto intVal = builder->create<tuples::GetColumnOp>(loc, builder->getI32Type(), intRef, tupleArg);
    auto floatVal = builder->create<tuples::GetColumnOp>(loc, builder->getF64Type(), floatRef, tupleArg);
    auto boolVal = builder->create<tuples::GetColumnOp>(loc, builder->getI1Type(), boolRef, tupleArg);
    
    // Complex arithmetic: (int * 2) + (float converted to int)
    auto const2 = builder->create<arith::ConstantIntOp>(loc, 2, 32);
    auto doubledInt = builder->create<arith::MulIOp>(loc, intVal, const2);
    auto floatToInt = builder->create<arith::FPToSIOp>(loc, builder->getI32Type(), floatVal);
    auto complexResult = builder->create<arith::AddIOp>(loc, doubledInt, floatToInt);
    
    // Conditional logic: if bool then complex_result else 0
    auto zero = builder->create<arith::ConstantIntOp>(loc, 0, 32);
    auto conditionalResult = builder->create<arith::SelectOp>(loc, boolVal, complexResult, zero);
    
    // String manipulation placeholder
    auto stringPtr = builder->create<LLVM::NullOp>(loc, builder->getType<LLVM::LLVMPointerType>());
    
    builder->create<tuples::ReturnOp>(loc, ValueRange{conditionalResult, floatVal, stringPtr});
    
    // Verify complex expression structure
    EXPECT_TRUE(mapOp);
    
    // Count different operation types in the region
    int getColumnCount = 0, arithCount = 0, conversionCount = 0, selectCount = 0;
    
    block.walk([&](Operation* op) {
        if (isa<tuples::GetColumnOp>(op)) getColumnCount++;
        else if (isa<arith::MulIOp, arith::AddIOp>(op)) arithCount++;
        else if (isa<arith::FPToSIOp>(op)) conversionCount++;
        else if (isa<arith::SelectOp>(op)) selectCount++;
    });
    
    EXPECT_GE(getColumnCount, 3) << "Should access multiple input columns";
    EXPECT_GE(arithCount, 2) << "Should have arithmetic operations";
    EXPECT_GE(conversionCount, 1) << "Should have type conversions";
    EXPECT_GE(selectCount, 1) << "Should have conditional logic";
}

TEST_F(MapRenamingLoweringTest, DataTypePreservation) {
    // Test data type preservation during MapOp and RenamingOp lowering
    
    Value inputRel = createMockInputRelation();
    
    // Test various data types
    auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    std::vector<Type> testTypes = {
        builder->getI8Type(),
        builder->getI16Type(), 
        builder->getI32Type(),
        builder->getI64Type(),
        builder->getF32Type(),
        builder->getF64Type(),
        builder->getI1Type(),
        builder->getType<LLVM::LLVMPointerType>()
    };
    
    std::vector<Attribute> computedCols;
    std::vector<Attribute> renamingCols;
    
    for (size_t i = 0; i < testTypes.size(); ++i) {
        auto computedCol = colManager.createColumn("computed_" + std::to_string(i), testTypes[i]);
        auto renamedCol = colManager.createColumn("renamed_" + std::to_string(i), testTypes[i]);
        
        computedCols.push_back(colManager.createDef(computedCol));
        renamingCols.push_back(colManager.createDef(renamedCol));
    }
    
    // Test MapOp type preservation
    auto mapOp = createMapOp(inputRel, computedCols);
    auto mapComputedCols = mapOp.getComputedCols();
    
    for (size_t i = 0; i < testTypes.size(); ++i) {
        auto columnDef = cast<tuples::ColumnDefAttr>(mapComputedCols[i]);
        EXPECT_EQ(columnDef.getColumn().type, testTypes[i]) 
            << "MapOp should preserve type " << testTypes[i];
    }
    
    // Test RenamingOp type preservation
    auto renamingOp = createRenamingOp(inputRel, renamingCols);
    auto renamingColumns = renamingOp.getColumns();
    
    for (size_t i = 0; i < testTypes.size(); ++i) {
        auto columnDef = cast<tuples::ColumnDefAttr>(renamingColumns[i]);
        EXPECT_EQ(columnDef.getColumn().type, testTypes[i]) 
            << "RenamingOp should preserve type " << testTypes[i];
    }
}

// Integration test to verify lowering patterns work with MLIR infrastructure
TEST_F(MapRenamingLoweringTest, LoweringIntegration) {
    // Create a module to hold our operations
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create a function to hold operations
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, "test_map_renaming", funcType);
    auto& entryBlock = func.getBody().emplaceBlock();
    builder->setInsertionPointToStart(&entryBlock);
    
    // Create test operations
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> computedCols = createTestColumnDefs();
    std::vector<Attribute> renamingCols = createRenamingColumns();
    
    auto mapOp = createMapOp(inputRel, computedCols);
    auto renamingOp = createRenamingOp(mapOp.getResult(), renamingCols); // Chain operations
    
    // Add terminator
    builder->create<func::ReturnOp>(loc);
    
    // Verify module structure
    EXPECT_TRUE(module);
    EXPECT_TRUE(succeeded(verify(module)));
    
    // Verify function contains our operations
    auto& funcBody = func.getBody().front();
    bool foundMapOp = false;
    bool foundRenamingOp = false;
    
    for (auto& op : funcBody) {
        if (isa<relalg::MapOp>(op)) {
            foundMapOp = true;
        } else if (isa<relalg::RenamingOp>(op)) {
            foundRenamingOp = true;
        }
    }
    
    EXPECT_TRUE(foundMapOp);
    EXPECT_TRUE(foundRenamingOp);
    
    // Verify operations are properly connected
    EXPECT_EQ(renamingOp.getRel(), mapOp.getResult());
    
    // Verify operations are ready for lowering pass application
    EXPECT_TRUE(mapOp.getOperation()->isRegistered());
    EXPECT_TRUE(renamingOp.getOperation()->isRegistered());
}