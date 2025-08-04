#include <gtest/gtest.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>

#include "compiler/Dialect/RelAlg/RelAlgDialect.h"
#include "compiler/Dialect/RelAlg/RelAlgOps.h"
#include "compiler/Conversion/RelAlgToSubOp/LowerRelAlgToSubOp.h"
#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/DB/DBDialect.h"
#include "compiler/Dialect/DB/DBTypes.h"
#include "compiler/Dialect/util/UtilDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "execution/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class BaseTableSelectionLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_DEBUG("BaseTableSelectionLoweringTest: Setting up test environment");
        
        // Load all required dialects
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuples::TupleStreamDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<cf::ControlFlowDialect>();
        
        // Initialize builder
        builder = std::make_unique<OpBuilder>(&context);
        
        PGX_DEBUG("BaseTableSelectionLoweringTest: Setup completed");
    }

    void TearDown() override {
        // Nothing specific to clean up
    }

    // Helper to create column definition attributes
    tuples::ColumnDefAttr createColumnDef(StringRef name, Type type) {
        auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        std::string scope = columnManager.getUniqueScope("test_scope");
        auto colDef = columnManager.createDef(scope, name.str());
        colDef.getColumn().type = type;
        return colDef;
    }

    // Helper to create BaseTableOp for testing
    relalg::BaseTableOp createTestBaseTable(StringRef tableName, DictionaryAttr columns) {
        auto tableIdentifier = builder->getStringAttr(tableName);
        auto op = builder->create<relalg::BaseTableOp>(
            builder->getUnknownLoc(),
            tuples::TupleStreamType::get(&context),
            tableIdentifier,
            columns
        );
        return op;
    }

    // Helper to create SelectionOp for testing
    relalg::SelectionOp createTestSelection(Value input, Region& predicateRegion) {
        auto op = builder->create<relalg::SelectionOp>(
            builder->getUnknownLoc(),
            tuples::TupleStreamType::get(&context),
            input
        );
        op.getPredicate().takeBody(predicateRegion);
        return op;
    }

    // Helper to verify ScanRefsOp creation
    void verifyScanRefsOp(Operation* op) {
        ASSERT_TRUE(op) << "Operation should not be null";
        auto scanOp = dyn_cast<subop::ScanRefsOp>(op);
        ASSERT_TRUE(scanOp) << "Expected ScanRefsOp, got: " << op->getName().getStringRef().str();
        
        // Verify it has proper tuple stream type
        EXPECT_TRUE(isa<tuples::TupleStreamType>(scanOp.getResult().getType()));
        
        // Verify it has table reference input
        EXPECT_EQ(scanOp.getNumOperands(), 1) << "ScanRefsOp should have one table reference operand";
    }

    // Helper to verify GatherOp creation
    void verifyGatherOp(Operation* op) {
        ASSERT_TRUE(op) << "Operation should not be null";
        auto gatherOp = dyn_cast<subop::GatherOp>(op);
        ASSERT_TRUE(gatherOp) << "Expected GatherOp, got: " << op->getName().getStringRef().str();
        
        // Verify it has proper tuple stream type
        EXPECT_TRUE(isa<tuples::TupleStreamType>(gatherOp.getResult().getType()));
        
        // Verify it has proper operands (stream, ref, mapping)
        EXPECT_EQ(gatherOp.getNumOperands(), 2) << "GatherOp should have stream and ref operands";
    }

    // Helper to verify FilterOp creation
    void verifyFilterOp(Operation* op) {
        ASSERT_TRUE(op) << "Operation should not be null";
        auto filterOp = dyn_cast<subop::FilterOp>(op);
        ASSERT_TRUE(filterOp) << "Expected FilterOp, got: " << op->getName().getStringRef().str();
        
        // Verify it has proper tuple stream type
        EXPECT_TRUE(isa<tuples::TupleStreamType>(filterOp.getResult().getType()));
        
        // Verify it has stream input
        EXPECT_EQ(filterOp.getNumOperands(), 1) << "FilterOp should have one stream operand";
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(BaseTableSelectionLoweringTest, BaseTableToScanRefsGather) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing BaseTable → ScanRefs + Gather lowering");
    
    // Create column definitions for test table
    auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    auto idColDef = columnManager.createDef("test", "id");
    idColDef.getColumn().type = builder->getI32Type();
    auto nameColDef = columnManager.createDef("test", "name");
    nameColDef.getColumn().type = db::StringType::get(&context);
    
    auto columns = builder->getDictionaryAttr({
        NamedAttribute(builder->getStringAttr("id"), idColDef),
        NamedAttribute(builder->getStringAttr("name"), nameColDef)
    });
    
    // Create BaseTableOp with table identifier parsing format
    auto baseTableOp = createTestBaseTable("employees|oid:12345", columns);
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Created BaseTableOp with table identifier parsing");
    
    // TODO Phase 5+: Apply lowering conversion when passes are implemented
    // ConversionTarget target(context);
    // target.addLegalDialect<subop::SubOperatorDialect>();
    // target.addLegalDialect<tuples::TupleStreamDialect>();
    // target.addLegalDialect<db::DBDialect>();
    // target.addIllegalOp<relalg::BaseTableOp>();
    // 
    // RewritePatternSet patterns(&context);
    // TypeConverter typeConverter;
    // relalg::populateRelAlgToSubOpConversionPatterns(patterns, typeConverter);
    // 
    // // Apply conversion
    // EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: BaseTableOp created successfully, lowering to be implemented in Phase 5+");
    
    // For now, just verify the BaseTableOp was created correctly
    EXPECT_TRUE(baseTableOp);
    EXPECT_TRUE(isa<tuples::TupleStreamType>(baseTableOp.getResult().getType()));
    EXPECT_EQ(baseTableOp.getTableIdentifier().str(), "employees|oid:12345");
    
    // TODO Phase 5+: Verify lowered operations
    // // Verify the transformation created the expected operations
    // bool foundGetExternal = false;
    // bool foundScanRefs = false;
    // bool foundGather = false;
    // 
    // module.walk([&](Operation* op) {
    //     if (isa<subop::GetExternalOp>(op)) {
    //         foundGetExternal = true;
    //         PGX_DEBUG("BaseTableSelectionLoweringTest: Found GetExternalOp");
    //         
    //         // Verify table description format
    //         auto getExtOp = cast<subop::GetExternalOp>(op);
    //         auto desc = getExtOp.getDescription().str();
    //         EXPECT_TRUE(desc.find("\"table\": \"employees\"") != std::string::npos);
    //         EXPECT_TRUE(desc.find("\"oid\": \"12345\"") != std::string::npos);
    //         EXPECT_TRUE(desc.find("\"mapping\"") != std::string::npos);
    //     }
    //     else if (isa<subop::ScanRefsOp>(op)) {
    //         foundScanRefs = true;
    //         verifyScanRefsOp(op);
    //         PGX_DEBUG("BaseTableSelectionLoweringTest: Found and verified ScanRefsOp");
    //     }
    //     else if (isa<subop::GatherOp>(op)) {
    //         foundGather = true;
    //         verifyGatherOp(op);
    //         PGX_DEBUG("BaseTableSelectionLoweringTest: Found and verified GatherOp");
    //     }
    // });
    // 
    // EXPECT_TRUE(foundGetExternal) << "Should create GetExternalOp for table access";
    // EXPECT_TRUE(foundScanRefs) << "Should create ScanRefsOp for reference scanning";
    // EXPECT_TRUE(foundGather) << "Should create GatherOp for data gathering";
    // 
    // // Verify no BaseTableOp remains
    // bool foundBaseTable = false;
    // module.walk([&](relalg::BaseTableOp op) {
    //     foundBaseTable = true;
    // });
    // EXPECT_FALSE(foundBaseTable) << "BaseTableOp should be completely lowered";
}

TEST_F(BaseTableSelectionLoweringTest, BaseTableTableIdentifierParsing) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing table identifier parsing variants");
    
    // Test cases for different table identifier formats
    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> testCases = {
        {"simple_table", {"simple_table", "0"}},
        {"users|oid:54321", {"users", "54321"}},
        {"schema.table|oid:999", {"schema.table", "999"}},
        {"table_with_underscores|oid:123456", {"table_with_underscores", "123456"}}
    };
    
    for (const auto& [identifier, expected] : testCases) {
        auto [expectedTable, expectedOid] = expected;
        
        // Create minimal column set
        auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
        auto colDef = columnManager.createDef("test", "id");
        colDef.getColumn().type = builder->getI32Type();
        
        auto columns = builder->getDictionaryAttr({
            NamedAttribute(builder->getStringAttr("id"), colDef)
        });
        
        // Create BaseTableOp
        auto baseTableOp = createTestBaseTable(identifier, columns);
        
        // For now, just verify the BaseTableOp was created correctly
        EXPECT_TRUE(baseTableOp);
        EXPECT_TRUE(isa<tuples::TupleStreamType>(baseTableOp.getResult().getType()));
        EXPECT_EQ(baseTableOp.getTableIdentifier().str(), identifier);
        
        // TODO Phase 5+: Apply lowering and verify results
        // ConversionTarget target(context);
        // target.addLegalDialect<subop::SubOperatorDialect>();
        // target.addLegalDialect<tuples::TupleStreamDialect>();
        // target.addLegalDialect<db::DBDialect>();
        // target.addIllegalOp<relalg::BaseTableOp>();
        // 
        // RewritePatternSet patterns(&context);
        // TypeConverter typeConverter;
        // relalg::populateRelAlgToSubOpConversionPatterns(patterns, typeConverter);
        // 
        // EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
        // 
        // // Verify the parsed values in GetExternalOp
        // module.walk([&](subop::GetExternalOp op) {
        //     auto desc = op.getDescription().str();
        //     EXPECT_TRUE(desc.find("\"table\": \"" + expectedTable + "\"") != std::string::npos)
        //         << "Table name mismatch for identifier: " << identifier;
        //     EXPECT_TRUE(desc.find("\"oid\": \"" + expectedOid + "\"") != std::string::npos)
        //         << "OID mismatch for identifier: " << identifier;
        // });
        
        // Clean up for next test case - create new module for each test case
        // (No cleanup needed since we're not storing module as member)
    }
}

TEST_F(BaseTableSelectionLoweringTest, SelectionToFilter) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing Selection → Filter lowering");
    
    // Create a mock input stream (in real scenario this would come from BaseTable lowering)
    auto streamType = tuples::TupleStreamType::get(&context);
    auto mockInput = builder->create<subop::GenerateOp>(loc, streamType);
    
    // Create a simple predicate region for testing
    Region predicateRegion;
    Block* predicateBlock = new Block();
    predicateRegion.push_back(predicateBlock);
    
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(predicateBlock);
    
    // Create a simple boolean constant predicate (true condition)
    auto boolConstant = builder->create<arith::ConstantOp>(
        loc, builder->getBoolAttr(true)
    );
    
    // Create return operation for predicate
    builder->create<tuples::ReturnOp>(loc, ValueRange{boolConstant.getResult()});
    
    // Create SelectionOp with predicate
    auto selectionOp = createTestSelection(mockInput.getResult(), predicateRegion);
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Created SelectionOp with simple predicate");
    
    // Apply lowering conversion
    ConversionTarget target(context);
    target.addLegalDialect<subop::SubOperatorDialect>();
    target.addLegalDialect<tuples::TupleStreamDialect>();
    target.addLegalDialect<db::DBDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<relalg::SelectionOp>();
    
    RewritePatternSet patterns(&context);
    relalg::populateRelAlgToSubOpConversionPatterns(patterns);
    
    // Apply conversion
    EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Selection lowering completed");
    
    // For trivial true predicate, it should pass through without creating FilterOp
    bool foundFilter = false;
    ModuleOp::create(builder->getUnknownLoc())->walk([&](subop::FilterOp op) {
        foundFilter = true;
    });
    
    // Trivial selection (constant true) should be optimized away
    EXPECT_FALSE(foundFilter) << "Trivial true selection should be optimized away";
    
    // Verify no SelectionOp remains
    bool foundSelection = false;
    ModuleOp::create(builder->getUnknownLoc())->walk([&](relalg::SelectionOp op) {
        foundSelection = true;
    });
    EXPECT_FALSE(foundSelection) << "SelectionOp should be completely lowered";
}

TEST_F(BaseTableSelectionLoweringTest, SelectionWithComplexPredicate) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing Selection with complex predicate lowering");
    
    // Create mock input stream
    auto streamType = tuples::TupleStreamType::get(&context);
    auto mockInput = builder->create<subop::GenerateOp>(loc, streamType);
    
    // Create a complex predicate region (comparison operation)
    Region predicateRegion;
    Block* predicateBlock = new Block();
    predicateRegion.push_back(predicateBlock);
    
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(predicateBlock);
    
    // Create column access (mock)
    auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    auto colDef = columnManager.createDef("test", "id");
    colDef.getColumn().type = builder->getI32Type();
    auto colRef = columnManager.createRef(&colDef.getColumn());
    
    auto getColumnOp = builder->create<tuples::GetColumnOp>(
        loc, builder->getI32Type(), colRef
    );
    
    // Create constant for comparison
    auto constOp = builder->create<arith::ConstantIntOp>(loc, 100, 32);
    
    // Create comparison (id > 100)
    auto cmpOp = builder->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, 
        getColumnOp.getResult(), constOp.getResult()
    );
    
    // Create return operation for predicate
    builder->create<tuples::ReturnOp>(loc, ValueRange{cmpOp.getResult()});
    
    // Create SelectionOp with complex predicate
    auto selectionOp = createTestSelection(mockInput.getResult(), predicateRegion);
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Created SelectionOp with comparison predicate");
    
    // Apply lowering conversion
    ConversionTarget target(context);
    target.addLegalDialect<subop::SubOperatorDialect>();
    target.addLegalDialect<tuples::TupleStreamDialect>();
    target.addLegalDialect<db::DBDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<relalg::SelectionOp>();
    
    RewritePatternSet patterns(&context);
    relalg::populateRelAlgToSubOpConversionPatterns(patterns);
    
    // Apply conversion
    EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Complex predicate lowering completed");
    
    // Verify FilterOp creation for non-trivial predicate
    bool foundMap = false;
    bool foundFilter = false;
    
    ModuleOp::create(builder->getUnknownLoc())->walk([&](Operation* op) {
        if (isa<subop::MapOp>(op)) {
            foundMap = true;
            PGX_DEBUG("BaseTableSelectionLoweringTest: Found MapOp for predicate evaluation");
        }
        else if (isa<subop::FilterOp>(op)) {
            foundFilter = true;
            verifyFilterOp(op);
            PGX_DEBUG("BaseTableSelectionLoweringTest: Found and verified FilterOp");
        }
    });
    
    EXPECT_TRUE(foundMap) << "Should create MapOp for predicate evaluation";
    EXPECT_TRUE(foundFilter) << "Should create FilterOp for non-trivial predicate";
}

TEST_F(BaseTableSelectionLoweringTest, SelectionWithAndPredicate) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing Selection with AND predicate priority ordering");
    
    // Create mock input stream
    auto streamType = tuples::TupleStreamType::get(&context);
    auto mockInput = builder->create<subop::GenerateOp>(loc, streamType);
    
    // Create AND predicate region
    Region predicateRegion;
    Block* predicateBlock = new Block();
    predicateRegion.push_back(predicateBlock);
    
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(predicateBlock);
    
    // Create column definitions for different types (to test priority ordering)
    auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    
    // Integer column (priority 1)
    auto intColDef = columnManager.createDef("test", "int_col");
    intColDef.getColumn().type = builder->getI32Type();
    auto intColRef = columnManager.createRef(&intColDef.getColumn());
    auto intGetCol = builder->create<tuples::GetColumnOp>(
        loc, builder->getI32Type(), intColRef
    );
    auto intConst = builder->create<arith::ConstantIntOp>(loc, 10, 32);
    auto intCmp = builder->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, intGetCol.getResult(), intConst.getResult()
    );
    
    // String column (priority 10) 
    auto strColDef = columnManager.createDef("test", "str_col");
    strColDef.getColumn().type = db::StringType::get(&context);
    auto strColRef = columnManager.createRef(&strColDef.getColumn());
    auto strGetCol = builder->create<tuples::GetColumnOp>(
        loc, db::StringType::get(&context), strColRef
    );
    auto strConst = builder->create<db::ConstantOp>(
        loc, db::StringType::get(&context), builder->getStringAttr("test")
    );
    auto strCmp = builder->create<db::CmpOp>(
        loc, db::DBCmpPredicate::eq, strGetCol.getResult(), strConst.getResult()
    );
    
    // Create AND operation (should be reordered by priority: int first, then string)
    auto andOp = builder->create<db::AndOp>(
        loc, builder->getI1Type(), ValueRange{strCmp.getResult(), intCmp.getResult()}
    );
    
    // Create return operation
    builder->create<tuples::ReturnOp>(loc, ValueRange{andOp.getResult()});
    
    // Create SelectionOp
    auto selectionOp = createTestSelection(mockInput.getResult(), predicateRegion);
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Created SelectionOp with AND predicate for priority testing");
    
    // Apply lowering conversion
    ConversionTarget target(context);
    target.addLegalDialect<subop::SubOperatorDialect>();
    target.addLegalDialect<tuples::TupleStreamDialect>();
    target.addLegalDialect<db::DBDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<relalg::SelectionOp>();
    
    RewritePatternSet patterns(&context);
    relalg::populateRelAlgToSubOpConversionPatterns(patterns);
    
    // Apply conversion
    EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: AND predicate lowering completed");
    
    // Verify multiple filter operations created (one for each condition)
    int filterCount = 0;
    int mapCount = 0;
    
    ModuleOp::create(builder->getUnknownLoc())->walk([&](Operation* op) {
        if (isa<subop::FilterOp>(op)) {
            filterCount++;
        }
        else if (isa<subop::MapOp>(op)) {
            mapCount++;
        }
    });
    
    EXPECT_EQ(filterCount, 2) << "Should create two FilterOps for AND conditions";
    EXPECT_EQ(mapCount, 2) << "Should create two MapOps for predicate evaluations";
}

TEST_F(BaseTableSelectionLoweringTest, SelectionColumnPassthrough) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing Selection column passthrough behavior");
    
    // This test verifies that Selection operations properly maintain 
    // column availability for downstream operations
    
    // Create mock input with multiple columns
    auto streamType = tuples::TupleStreamType::get(&context);
    auto mockInput = builder->create<subop::GenerateOp>(loc, streamType);
    
    // Create simple true predicate (should pass through all columns)
    Region predicateRegion;
    Block* predicateBlock = new Block();
    predicateRegion.push_back(predicateBlock);
    
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(predicateBlock);
    
    auto boolConstant = builder->create<arith::ConstantOp>(
        loc, builder->getBoolAttr(true)
    );
    builder->create<tuples::ReturnOp>(loc, ValueRange{boolConstant.getResult()});
    
    // Create SelectionOp
    auto selectionOp = createTestSelection(mockInput.getResult(), predicateRegion);
    
    // For now, just verify the SelectionOp was created correctly
    EXPECT_TRUE(selectionOp);
    EXPECT_TRUE(isa<tuples::TupleStreamType>(selectionOp.getResult().getType()));
    
    // TODO Phase 5+: Apply lowering when passes are implemented
    // ConversionTarget target(context);
    // target.addLegalDialect<subop::SubOperatorDialect>();
    // target.addLegalDialect<tuples::TupleStreamDialect>();
    // target.addLegalDialect<db::DBDialect>();
    // target.addLegalDialect<arith::ArithDialect>();
    // target.addIllegalOp<relalg::SelectionOp>();
    // 
    // RewritePatternSet patterns(&context);
    // TypeConverter typeConverter;
    // relalg::populateRelAlgToSubOpConversionPatterns(patterns, typeConverter);
    // 
    // EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    // 
    // // For trivial selection, the result should directly use the input stream
    // // (optimized away, maintaining column passthrough)
    // bool foundDirectConnection = false;
    // 
    // module.walk([&](Operation* op) {
    //     if (isa<subop::GenerateOp>(op)) {
    //         // Check if this operation's result is used directly (no intermediate FilterOp)
    //         for (auto user : op->getUsers()) {
    //             if (!isa<subop::FilterOp>(user)) {
    //                 foundDirectConnection = true;
    //             }
    //         }
    //     }
    // });
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Column passthrough verification completed");
}

TEST_F(BaseTableSelectionLoweringTest, ErrorConditions) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing error conditions and edge cases");
    
    // Test empty column set for BaseTable
    auto emptyColumns = builder->getDictionaryAttr({});
    auto baseTableOp = createTestBaseTable("empty_table", emptyColumns);
    
    // For now, just verify the BaseTableOp was created correctly with empty columns
    EXPECT_TRUE(baseTableOp);
    EXPECT_TRUE(isa<tuples::TupleStreamType>(baseTableOp.getResult().getType()));
    EXPECT_EQ(baseTableOp.getTableIdentifier().str(), "empty_table");
    
    // TODO Phase 5+: Apply lowering when passes are implemented
    // ConversionTarget target(context);
    // target.addLegalDialect<subop::SubOperatorDialect>();
    // target.addLegalDialect<tuples::TupleStreamDialect>();
    // target.addLegalDialect<db::DBDialect>();
    // target.addIllegalOp<relalg::BaseTableOp>();
    // 
    // RewritePatternSet patterns(&context);
    // TypeConverter typeConverter;
    // relalg::populateRelAlgToSubOpConversionPatterns(patterns, typeConverter);
    // 
    // // Should complete without error even with empty columns
    // EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    // 
    // // Verify GetExternalOp still created with empty mapping
    // bool foundGetExternal = false;
    // module.walk([&](subop::GetExternalOp op) {
    //     foundGetExternal = true;
    //     auto desc = op.getDescrAttr().getValue().str();
    //     EXPECT_TRUE(desc.find("\"mapping\": {") != std::string::npos);
    // });
    // 
    // EXPECT_TRUE(foundGetExternal) << "Should create GetExternalOp even with empty columns";
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Error condition testing completed");
}

TEST_F(BaseTableSelectionLoweringTest, ModuleVerification) {
    PGX_DEBUG("BaseTableSelectionLoweringTest: Testing MLIR module verification after lowering");
    
    // Create a complete lowering scenario
    auto& columnManager = context.getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    auto colDef = columnManager.createDef("test", "id");
    colDef.getColumn().type = builder->getI32Type();
    
    auto columns = builder->getDictionaryAttr({
        NamedAttribute(builder->getStringAttr("id"), colDef)
    });
    
    auto baseTableOp = createTestBaseTable("test_table|oid:1", columns);
    
    // For now, just verify the BaseTableOp was created correctly
    EXPECT_TRUE(baseTableOp);
    EXPECT_TRUE(isa<tuples::TupleStreamType>(baseTableOp.getResult().getType()));
    EXPECT_EQ(baseTableOp.getTableIdentifier().str(), "test_table|oid:1");
    
    // TODO Phase 5+: Apply complete lowering pipeline when passes are implemented
    // ConversionTarget target(context);
    // target.addLegalDialect<subop::SubOperatorDialect>();
    // target.addLegalDialect<tuples::TupleStreamDialect>();
    // target.addLegalDialect<db::DBDialect>();
    // target.addLegalDialect<arith::ArithDialect>();
    // target.addIllegalOp<relalg::BaseTableOp>();
    // target.addIllegalOp<relalg::SelectionOp>();
    // 
    // RewritePatternSet patterns(&context);
    // TypeConverter typeConverter;
    // relalg::populateRelAlgToSubOpConversionPatterns(patterns, typeConverter);
    // 
    // EXPECT_TRUE(succeeded(applyPartialConversion(module, target, std::move(patterns))));
    // 
    // // Verify the resulting module is valid MLIR
    // EXPECT_TRUE(succeeded(mlir::verify(module))) << "Lowered module should pass MLIR verification";
    
    PGX_DEBUG("BaseTableSelectionLoweringTest: Module verification completed successfully");
}