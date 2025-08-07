#include <gtest/gtest.h>
#include "execution/logging.h"

#include "mlir/Conversion/RelAlgToDSA/RelAlgToDSA.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class RelAlgToDSALoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        
        PGX_DEBUG("RelAlgToDSALoweringTest setup completed");
    }

    // Helper function to run the lowering pass on a module
    LogicalResult runLoweringPass(ModuleOp module) {
        PGX_DEBUG("runLoweringPass: Creating conversion target...");
        // Create the conversion target
        ConversionTarget target(context);
        
        PGX_DEBUG("runLoweringPass: Adding legal dialects...");
        // DSA dialect is legal
        target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        PGX_DEBUG("runLoweringPass: Adding illegal operations...");
        // RelAlg operations are illegal (need to be converted)
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        target.addIllegalOp<::pgx::mlir::relalg::MaterializeOp>();
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();
        
        PGX_DEBUG("runLoweringPass: Creating rewrite patterns...");
        // Create the rewrite patterns
        RewritePatternSet patterns(&context);
        
        PGX_DEBUG("runLoweringPass: Adding conversion patterns...");
        // Add conversion patterns (matching the pass implementation)
        patterns.add<mlir::pgx_conversion::BaseTableToScanSourcePattern>(&context);
        patterns.add<mlir::pgx_conversion::MaterializeToResultBuilderPattern>(&context);
        patterns.add<mlir::pgx_conversion::ReturnOpLoweringPattern>(&context);
        
        PGX_DEBUG("runLoweringPass: About to apply partial conversion...");
        // Apply the conversion to the entire module
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            PGX_ERROR("runLoweringPass: applyPartialConversion failed");
            return failure();
        }
        
        PGX_DEBUG("runLoweringPass: Conversion completed successfully");
        return success();
    }

    MLIRContext context;
    OpBuilder builder{&context};
};

TEST_F(RelAlgToDSALoweringTest, BaseTableOpToScanSourceOpLowering) {
    PGX_DEBUG("Testing actual BaseTableOp to ScanSourceOp lowering");
    
    // Create a module with BaseTableOp
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_base_table_lowering", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp with specific table name and OID
    const int64_t testOid = 12345;
    const std::string tableName = "test_table";
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr(tableName),
        builder.getI64IntegerAttr(testOid));
    
    // Initialize BaseTableOp region properly
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify BaseTableOp exists before lowering
    bool foundBaseTableBefore = false;
    func.walk([&](::pgx::mlir::relalg::BaseTableOp op) {
        foundBaseTableBefore = true;
        EXPECT_EQ(op.getTableName(), tableName);
        EXPECT_EQ(op.getTableOid(), testOid);
    });
    EXPECT_TRUE(foundBaseTableBefore) << "BaseTableOp should exist before lowering";
    
    // Run the lowering pass
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Lowering pass should succeed";
    
    // Verify BaseTableOp was converted to ScanSourceOp
    bool foundBaseTableAfter = false;
    bool foundScanSourceAfter = false;
    std::string scanSourceJSON;
    
    func.walk([&](::pgx::mlir::relalg::BaseTableOp op) {
        foundBaseTableAfter = true;
    });
    
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp op) {
        foundScanSourceAfter = true;
        scanSourceJSON = op.getTableDescription().str();
    });
    
    // Verify the conversion worked correctly
    EXPECT_FALSE(foundBaseTableAfter) << "BaseTableOp should be removed after lowering";
    EXPECT_TRUE(foundScanSourceAfter) << "ScanSourceOp should be created after lowering";
    
    // Verify the JSON contains the correct table information
    EXPECT_NE(scanSourceJSON.find("\"table\":\"" + tableName + "\""), std::string::npos)
        << "JSON should contain table name: " << scanSourceJSON;
    EXPECT_NE(scanSourceJSON.find("\"oid\":" + std::to_string(testOid)), std::string::npos)
        << "JSON should contain table OID: " << scanSourceJSON;
    
    PGX_DEBUG("BaseTableOp to ScanSourceOp lowering test completed successfully");
}

TEST_F(RelAlgToDSALoweringTest, MaterializeOpLoweringValidation) {
    PGX_DEBUG("DEBUGGING: Testing minimal MaterializeOp creation to isolate the segfault");
    
    // STEP 1: Test only MaterializeOp creation without lowering
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_materialize_debug", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    PGX_DEBUG("DEBUGGING: About to create BaseTableOp");
    
    // Create BaseTableOp as source for MaterializeOp
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    PGX_DEBUG("DEBUGGING: BaseTableOp created successfully");
    
    // Initialize BaseTableOp region properly
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    PGX_DEBUG("DEBUGGING: BaseTableOp region initialized");
    
    // Create MaterializeOp with columns
    builder.setInsertionPointAfter(baseTableOp);
    SmallVector<Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("id"));
    columnAttrs.push_back(builder.getStringAttr("name"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    PGX_DEBUG("DEBUGGING: About to create MaterializeOp");
    
    auto materializeOp = builder.create<::pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    PGX_DEBUG("DEBUGGING: MaterializeOp created successfully");
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    PGX_DEBUG("DEBUGGING: Function completed, module should be valid");
    
    // Verify MaterializeOp exists before attempting lowering
    bool foundMaterialize = false;
    func.walk([&](::pgx::mlir::relalg::MaterializeOp op) {
        foundMaterialize = true;
    });
    EXPECT_TRUE(foundMaterialize) << "MaterializeOp should exist before lowering";
    
    PGX_DEBUG("DEBUGGING: Module walk completed successfully");
    
    // STEP 2: Now test the lowering pass execution
    PGX_DEBUG("DEBUGGING: About to run lowering pass");
    
    // Run the lowering pass
    LogicalResult result = runLoweringPass(module);
    
    if (succeeded(result)) {
        PGX_DEBUG("DEBUGGING: Lowering pass completed successfully!");
        
        // Verify the conversion worked correctly
        bool foundMaterialize = false;
        bool foundCreateDS = false;
        bool foundForOp = false;
        bool foundFinalizeOp = false;
        
        func.walk([&](Operation* op) {
            if (llvm::isa<::pgx::mlir::relalg::MaterializeOp>(op)) {
                foundMaterialize = true;
            } else if (llvm::isa<::pgx::mlir::dsa::CreateDSOp>(op)) {
                foundCreateDS = true;
            } else if (llvm::isa<::pgx::mlir::dsa::ForOp>(op)) {
                foundForOp = true;
            } else if (llvm::isa<::pgx::mlir::dsa::FinalizeOp>(op)) {
                foundFinalizeOp = true;
            }
        });
        
        // Verify the conversion worked correctly
        EXPECT_FALSE(foundMaterialize) << "MaterializeOp should be removed after lowering";
        EXPECT_TRUE(foundCreateDS) << "CreateDSOp should be created for result builder";
        EXPECT_TRUE(foundForOp) << "ForOp should be created for iteration";
        EXPECT_TRUE(foundFinalizeOp) << "FinalizeOp should be created for final result";
        
        PGX_DEBUG("DEBUGGING: All verification checks passed!");
    } else {
        PGX_ERROR("DEBUGGING: Lowering pass failed");
        FAIL() << "Lowering pass should succeed";
    }
}

TEST_F(RelAlgToDSALoweringTest, BaseTableOpReplacesReturnOp) {
    PGX_DEBUG("Testing that BaseTableOp lowering correctly replaces entire operation including ReturnOp");
    
    // Create a module with BaseTableOp containing ReturnOp
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_basetable_replacement", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp with ReturnOp in its region
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("test_returns"),
        builder.getI64IntegerAttr(111));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    auto returnOp = builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify starting state
    bool foundBaseTableBefore = false;
    bool foundReturnBefore = false;
    func.walk([&](::pgx::mlir::relalg::BaseTableOp op) {
        foundBaseTableBefore = true;
    });
    func.walk([&](::pgx::mlir::relalg::ReturnOp op) {
        foundReturnBefore = true;
    });
    EXPECT_TRUE(foundBaseTableBefore) << "BaseTableOp should exist before lowering";
    EXPECT_TRUE(foundReturnBefore) << "ReturnOp should exist before lowering";
    
    // Run the lowering pass
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Lowering pass should succeed";
    
    // Verify BaseTableOp (and its contained ReturnOp) was replaced by ScanSourceOp
    bool foundBaseTableAfter = false;
    bool foundReturnAfter = false;
    bool foundScanSourceAfter = false;
    
    func.walk([&](::pgx::mlir::relalg::BaseTableOp op) {
        foundBaseTableAfter = true;
    });
    
    func.walk([&](::pgx::mlir::relalg::ReturnOp op) {
        foundReturnAfter = true;
    });
    
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp op) {
        foundScanSourceAfter = true;
    });
    
    // CORRECT EXPECTATION: BaseTableOp replacement removes the entire operation including its region
    EXPECT_FALSE(foundBaseTableAfter) << "BaseTableOp should be replaced after lowering";
    EXPECT_FALSE(foundReturnAfter) << "ReturnOp should be removed with its parent BaseTableOp";
    EXPECT_TRUE(foundScanSourceAfter) << "ScanSourceOp should be created to replace BaseTableOp";
    
    PGX_DEBUG("BaseTableOp replacement test completed successfully - region is correctly discarded");
}

TEST_F(RelAlgToDSALoweringTest, FullPipelineSelectStarFromTest) {
    // RE-ENABLED - MaterializeOp lowering should now work with proper nested ForOp pattern
    // Test with MaterializeOp to identify the exact issue
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "select_star_from_test", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(16384));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    // Add MaterializeOp - this is what causes the segfault
    builder.setInsertionPointAfter(baseTableOp);
    SmallVector<Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("id"));
    columnAttrs.push_back(builder.getStringAttr("name"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<::pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run the lowering pass - this should trigger the segfault
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Full lowering should succeed";
    
    // Verify that BaseTableOp was converted to ScanSourceOp
    bool foundScanSource = false;
    bool foundBaseTable = false;
    
    func.walk([&](Operation* op) {
        if (llvm::isa<::pgx::mlir::dsa::ScanSourceOp>(op)) {
            foundScanSource = true;
        }
        if (llvm::isa<::pgx::mlir::relalg::BaseTableOp>(op)) {
            foundBaseTable = true;
        }
    });
    
    // Basic verification
    EXPECT_TRUE(foundScanSource) << "Should have ScanSourceOp after lowering";
    EXPECT_FALSE(foundBaseTable) << "BaseTableOp should be replaced after lowering";
    
    PGX_DEBUG("Simplified pipeline test completed successfully");
}

TEST_F(RelAlgToDSALoweringTest, VerifyDSAStructure) {
    PGX_DEBUG("Testing that lowered DSA IR has correct structure");
    
    // Create a comprehensive test case
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "verify_dsa_structure", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create RelAlg operations
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("orders"),
        builder.getI64IntegerAttr(99999));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    SmallVector<Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("order_id"));
    columnAttrs.push_back(builder.getStringAttr("customer_id"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<::pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run lowering
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Lowering should succeed";
    
    // Verify DSA operations have correct types and structure
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp scanOp) {
        auto resultType = scanOp.getResult().getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::GenericIterableType>(resultType))
            << "ScanSourceOp should return GenericIterableType";
    });
    
    func.walk([&](::pgx::mlir::dsa::CreateDSOp createOp) {
        auto resultType = createOp.getResult().getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::TableBuilderType>(resultType))
            << "CreateDSOp should return TableBuilderType";
    });
    
    func.walk([&](::pgx::mlir::dsa::FinalizeOp finalizeOp) {
        auto resultType = finalizeOp.getResult().getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::TableType>(resultType))
            << "FinalizeOp should return TableType";
    });
    
    // Verify nested ForOp structure - should have exactly 2 ForOps (outer and inner)
    int forOpCount = 0;
    bool foundOuterForOp = false;
    bool foundInnerForOp = false;
    
    func.walk([&](::pgx::mlir::dsa::ForOp forOp) {
        forOpCount++;
        
        // Verify ForOp has proper block structure
        EXPECT_EQ(forOp.getBody().getBlocks().size(), 1)
            << "ForOp should have exactly one block";
        
        Block &forBlock = forOp.getBody().front();
        EXPECT_EQ(forBlock.getNumArguments(), 1)
            << "ForOp block should have exactly one argument";
        
        auto argType = forBlock.getArgument(0).getType();
        
        // Check if this is outer ForOp (processes RecordBatch) or inner ForOp (processes individual Record)
        if (llvm::isa<::pgx::mlir::dsa::RecordBatchType>(argType)) {
            foundOuterForOp = true;
            PGX_DEBUG("Found outer ForOp with RecordBatchType argument");
        } else if (llvm::isa<::pgx::mlir::dsa::RecordType>(argType)) {
            foundInnerForOp = true;
            PGX_DEBUG("Found inner ForOp with RecordType argument");
        }
    });
    
    // Verify nested ForOp structure
    EXPECT_EQ(forOpCount, 2) << "Should have exactly 2 ForOps (outer and inner)";
    EXPECT_TRUE(foundOuterForOp) << "Should have outer ForOp processing RecordBatch";
    EXPECT_TRUE(foundInnerForOp) << "Should have inner ForOp processing individual Records";
    
    func.walk([&](::pgx::mlir::dsa::AtOp atOp) {
        auto recordType = atOp.getRecord().getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::RecordType>(recordType))
            << "AtOp record operand should be RecordType";
        
        auto columnAttr = atOp.getColumnNameAttr();
        EXPECT_TRUE(columnAttr) << "AtOp should have column name attribute";
        
        std::string columnName = columnAttr.getValue().str();
        EXPECT_TRUE(columnName == "order_id" || columnName == "customer_id")
            << "AtOp column name should match expected: " << columnName;
    });
    
    PGX_DEBUG("DSA structure verification completed successfully");
}