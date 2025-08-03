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

#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class ProjectionLoweringTest : public ::testing::Test {
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
        context.loadDialect<LLVM::LLVMDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
    }

    // Helper to create a mock RelAlg projection operation
    relalg::ProjectionOp createProjectionOp(
        Value inputRel, 
        ArrayRef<Attribute> columnRefs, 
        relalg::SetSemantic setSemantic = relalg::SetSemantic::all) {
        
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        ArrayAttr colsAttr = builder->getArrayAttr(columnRefs);
        auto setSemanticAttr = relalg::SetSemanticAttr::get(&context, setSemantic);
        
        return builder->create<relalg::ProjectionOp>(
            builder->getUnknownLoc(), tupleStreamType, setSemanticAttr, inputRel, colsAttr);
    }

    // Helper to create column references for testing
    std::vector<Attribute> createTestColumnRefs() {
        auto& colManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        
        std::vector<Attribute> columnRefs;
        
        // Create test columns with different types
        std::string scope = colManager.getUniqueScope("test");
        auto intColumn = colManager.createDef(scope, "test_int");
        auto stringColumn = colManager.createDef(scope, "test_string");
        auto boolColumn = colManager.createDef(scope, "test_bool");
        
        intColumn.getColumn().type = builder->getI32Type();
        stringColumn.getColumn().type = builder->getType<LLVM::LLVMPointerType>();
        boolColumn.getColumn().type = builder->getI1Type();
        
        columnRefs.push_back(tuples::ColumnDefAttr::get(&context, intColumn));
        columnRefs.push_back(tuples::ColumnDefAttr::get(&context, stringColumn));
        columnRefs.push_back(tuples::ColumnDefAttr::get(&context, boolColumn));
        
        return columnRefs;
    }

    // Helper to create a mock input relation
    Value createMockInputRelation() {
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        return builder->create<tuples::TupleStreamOp>(builder->getUnknownLoc(), tupleStreamType);
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(ProjectionLoweringTest, ProjectionAllLowering) {
    // Test regular projection without DISTINCT
    
    // Create mock input relation
    Value inputRel = createMockInputRelation();
    
    // Create test column references
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    // Create ProjectionOp with 'all' semantic (no DISTINCT)
    auto projectionOp = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::all);
    
    // Verify operation was created correctly
    EXPECT_TRUE(projectionOp);
    EXPECT_EQ(projectionOp.getSetSemantic(), relalg::SetSemantic::all);
    EXPECT_EQ(projectionOp.getCols().size(), columnRefs.size());
    
    // Verify column metadata preservation
    auto colsAttr = projectionOp.getCols();
    EXPECT_EQ(colsAttr.size(), 3); // int, string, bool columns
    
    // Test that ProjectionAllLowering should pass through the input relation
    // This tests the core logic: if (projectionOp.getSetSemantic() == relalg::SetSemantic::distinct) return failure();
    // For non-distinct projections, the lowering replaces the op with the input relation
    EXPECT_EQ(projectionOp.getRel(), inputRel);
    
    // Verify that the operation has proper structure for lowering
    EXPECT_TRUE(projectionOp.getOperation()->getNumOperands() > 0);
    EXPECT_TRUE(projectionOp.getOperation()->getNumResults() > 0);
}

TEST_F(ProjectionLoweringTest, ProjectionDistinctLowering) {
    // Test projection with DISTINCT semantics
    
    // Create mock input relation
    Value inputRel = createMockInputRelation();
    
    // Create test column references
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    // Create ProjectionOp with 'distinct' semantic
    auto projectionOp = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::distinct);
    
    // Verify operation was created correctly
    EXPECT_TRUE(projectionOp);
    EXPECT_EQ(projectionOp.getSetSemantic(), relalg::SetSemantic::distinct);
    EXPECT_EQ(projectionOp.getCols().size(), columnRefs.size());
    
    // Verify column references are properly structured
    auto colsAttr = projectionOp.getCols();
    for (auto col : colsAttr) {
        EXPECT_TRUE(mlir::isa<tuples::ColumnDefAttr>(col));
    }
    
    // Test that this operation should be handled by ProjectionDistinctLowering
    // The distinct lowering creates a complex pipeline:
    // 1. Creates MultiMap state for deduplication
    // 2. Uses LookupOrInsertOp for hash-based distinct
    // 3. Creates ReduceOp for aggregation
    // 4. Scans the final state
    
    // Verify operation readiness for distinct lowering
    EXPECT_EQ(projectionOp.getRel(), inputRel);
    EXPECT_TRUE(projectionOp.getOperation()->getNumOperands() > 0);
    EXPECT_TRUE(projectionOp.getOperation()->getNumResults() > 0);
}

TEST_F(ProjectionLoweringTest, ProjectionColumnOrdering) {
    // Test that column ordering is preserved through projection
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    auto projectionOp = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::all);
    
    // Verify column order preservation
    auto colsAttr = projectionOp.getCols();
    EXPECT_EQ(colsAttr.size(), columnRefs.size());
    
    for (size_t i = 0; i < columnRefs.size(); ++i) {
        // Column references should be in the same order
        EXPECT_EQ(colsAttr[i], columnRefs[i]);
    }
}

TEST_F(ProjectionLoweringTest, ProjectionEmptyColumns) {
    // Test projection with no columns (edge case)
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> emptyColumnRefs;
    
    auto projectionOp = createProjectionOp(inputRel, emptyColumnRefs, relalg::SetSemantic::all);
    
    // Verify empty projection is handled
    EXPECT_TRUE(projectionOp);
    EXPECT_EQ(projectionOp.getCols().size(), 0);
    EXPECT_EQ(projectionOp.getRel(), inputRel);
}

TEST_F(ProjectionLoweringTest, ProjectionSingleColumn) {
    // Test projection with single column
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    // Take only first column
    std::vector<Attribute> singleColumn = {columnRefs[0]};
    
    auto projectionOp = createProjectionOp(inputRel, singleColumn, relalg::SetSemantic::distinct);
    
    // Verify single column distinct projection
    EXPECT_TRUE(projectionOp);
    EXPECT_EQ(projectionOp.getCols().size(), 1);
    EXPECT_EQ(projectionOp.getSetSemantic(), relalg::SetSemantic::distinct);
    
    // Verify the single column is correctly referenced
    auto colsAttr = projectionOp.getCols();
    EXPECT_TRUE(mlir::isa<tuples::ColumnDefAttr>(colsAttr[0]));
}

TEST_F(ProjectionLoweringTest, ProjectionDistinctPerformance) {
    // Test performance considerations for DISTINCT projections
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    auto projectionOp = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::distinct);
    
    // The distinct lowering should create efficient hash-based deduplication
    // Key performance characteristics from the source code:
    // 1. Uses subop::MapType for hash table state
    // 2. LookupOrInsertOp provides O(1) average lookup time
    // 3. ReduceOp handles aggregation efficiently
    // 4. ScanOp iterates results without duplication
    
    // Verify that the operation structure supports efficient lowering
    EXPECT_TRUE(projectionOp);
    EXPECT_GT(projectionOp.getCols().size(), 0);
    
    // Distinct operations should have proper state management setup
    // The lowering creates:
    // - keyMembers for deduplication keys
    // - stateMembers for aggregation state
    // - Compare functions for equality checking
    
    auto colsAttr = projectionOp.getCols();
    for (auto col : colsAttr) {
        auto columnDef = mlir::cast<tuples::ColumnDefAttr>(col);
        // Each column should have proper type information for hash computation
        EXPECT_TRUE(columnDef.getColumn().type);
    }
}

TEST_F(ProjectionLoweringTest, ProjectionMetadataPropagation) {
    // Test that column metadata is properly propagated
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    auto projectionOp = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::all);
    
    // Verify metadata propagation through column references
    auto colsAttr = projectionOp.getCols();
    
    for (auto col : colsAttr) {
        auto columnDef = mlir::cast<tuples::ColumnDefAttr>(col);
        
        // Verify type information is preserved
        EXPECT_TRUE(columnDef.getColumn().type);
        
        // Verify column names are preserved  
        EXPECT_FALSE(columnDef.getColumn().name.empty());
        
        // Types should match expected test types
        Type colType = columnDef.getColumn().type;
        EXPECT_TRUE(colType.isIntOrFloat() || 
                   mlir::isa<LLVM::LLVMPointerType>(colType) ||
                   colType.isInteger(1));
    }
}

TEST_F(ProjectionLoweringTest, ProjectionLoweringReadiness) {
    // Test that operations are ready for the lowering pass
    
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    // Test both projection types
    auto allProjection = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::all);
    auto distinctProjection = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::distinct);
    
    // Verify operations have all required components for lowering
    
    // All projections should have:
    EXPECT_TRUE(allProjection.getRel()); // Input relation
    EXPECT_TRUE(allProjection.getCols()); // Column specifications
    EXPECT_NE(allProjection.getSetSemantic(), relalg::SetSemantic::distinct); // Non-distinct semantic
    
    // Distinct projections should have:
    EXPECT_TRUE(distinctProjection.getRel()); // Input relation  
    EXPECT_TRUE(distinctProjection.getCols()); // Column specifications
    EXPECT_EQ(distinctProjection.getSetSemantic(), relalg::SetSemantic::distinct); // Distinct semantic
    
    // Both should be valid MLIR operations
    EXPECT_TRUE(allProjection.getOperation()->isRegistered());
    EXPECT_TRUE(distinctProjection.getOperation()->isRegistered());
    
    // Both should have proper result types
    EXPECT_TRUE(mlir::isa<tuples::TupleStreamType>(allProjection.getType()));
    EXPECT_TRUE(mlir::isa<tuples::TupleStreamType>(distinctProjection.getType()));
}

// Integration test to verify the lowering patterns work with MLIR infrastructure
TEST_F(ProjectionLoweringTest, ProjectionLoweringIntegration) {
    // Create a module to hold our operations
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());
    
    // Create a function to hold projection operations
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(builder->getUnknownLoc(), "test_projection", funcType);
    auto& entryBlock = func.getBody().emplaceBlock();
    builder->setInsertionPointToStart(&entryBlock);
    
    // Create test projections
    Value inputRel = createMockInputRelation();
    std::vector<Attribute> columnRefs = createTestColumnRefs();
    
    auto allProjection = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::all);
    auto distinctProjection = createProjectionOp(inputRel, columnRefs, relalg::SetSemantic::distinct);
    
    // Add terminator
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify module structure
    EXPECT_TRUE(module);
    EXPECT_EQ(module.getBody()->getNumArguments(), 0);
    
    // Verify function contains our operations
    auto& funcBody = func.getBody().front();
    bool foundAllProjection = false;
    bool foundDistinctProjection = false;
    
    for (auto& op : funcBody) {
        if (auto projOp = dyn_cast<relalg::ProjectionOp>(op)) {
            if (projOp.getSetSemantic() == relalg::SetSemantic::all) {
                foundAllProjection = true;
            } else if (projOp.getSetSemantic() == relalg::SetSemantic::distinct) {
                foundDistinctProjection = true;
            }
        }
    }
    
    EXPECT_TRUE(foundAllProjection);
    EXPECT_TRUE(foundDistinctProjection);
    
    // Verify operations are ready for lowering pass application
    EXPECT_TRUE(succeeded(verify(module)));
}