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
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        
        PGX_DEBUG("RelAlgToDSALoweringTest setup completed");
    }

    // Helper function to run the lowering pass on a module
    LogicalResult runLoweringPass(ModuleOp module) {
        // Create the conversion target
        ConversionTarget target(context);
        
        // DSA dialect is legal
        target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        // RelAlg operations are illegal (need to be converted)
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        target.addIllegalOp<::pgx::mlir::relalg::MaterializeOp>();
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();
        
        // Create the rewrite patterns
        RewritePatternSet patterns(&context);
        
        // Add conversion patterns (matching the pass implementation)
        patterns.add<mlir::pgx_conversion::BaseTableToScanSourcePattern>(&context);
        patterns.add<mlir::pgx_conversion::MaterializeToResultBuilderPattern>(&context);
        patterns.add<mlir::pgx_conversion::ReturnOpLoweringPattern>(&context);
        
        // Apply the conversion to the entire module
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            return failure();
        }
        
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

TEST_F(RelAlgToDSALoweringTest, MaterializeOpToDSAPatternLowering) {
    PGX_DEBUG("Testing actual MaterializeOp to DSA pattern lowering");
    
    // Create a module with BaseTableOp + MaterializeOp
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_materialize_lowering", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp as source
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("employees"),
        builder.getI64IntegerAttr(67890));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    // Create MaterializeOp with specific columns
    builder.setInsertionPointAfter(baseTableOp);
    SmallVector<Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("emp_id"));
    columnAttrs.push_back(builder.getStringAttr("emp_name"));
    columnAttrs.push_back(builder.getStringAttr("salary"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<::pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify MaterializeOp exists before lowering
    bool foundMaterializeBefore = false;
    func.walk([&](::pgx::mlir::relalg::MaterializeOp op) {
        foundMaterializeBefore = true;
        EXPECT_EQ(op.getColumns().size(), 3);
    });
    EXPECT_TRUE(foundMaterializeBefore) << "MaterializeOp should exist before lowering";
    
    // Run the lowering pass
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Lowering pass should succeed";
    
    // Verify MaterializeOp was converted to DSA pattern
    bool foundMaterializeAfter = false;
    bool foundScanSource = false;
    bool foundCreateDS = false;
    bool foundForOp = false;
    bool foundFinalizeOp = false;
    int atOpCount = 0;
    int dsAppendCount = 0;
    int nextRowCount = 0;
    
    func.walk([&](Operation* op) {
        if (llvm::isa<::pgx::mlir::relalg::MaterializeOp>(op)) {
            foundMaterializeAfter = true;
        } else if (llvm::isa<::pgx::mlir::dsa::ScanSourceOp>(op)) {
            foundScanSource = true;
        } else if (llvm::isa<::pgx::mlir::dsa::CreateDSOp>(op)) {
            foundCreateDS = true;
        } else if (llvm::isa<::pgx::mlir::dsa::ForOp>(op)) {
            foundForOp = true;
        } else if (llvm::isa<::pgx::mlir::dsa::FinalizeOp>(op)) {
            foundFinalizeOp = true;
        } else if (llvm::isa<::pgx::mlir::dsa::AtOp>(op)) {
            atOpCount++;
        } else if (llvm::isa<::pgx::mlir::dsa::DSAppendOp>(op)) {
            dsAppendCount++;
        } else if (llvm::isa<::pgx::mlir::dsa::NextRowOp>(op)) {
            nextRowCount++;
        }
    });
    
    // Verify the complete DSA pattern was created
    EXPECT_FALSE(foundMaterializeAfter) << "MaterializeOp should be removed after lowering";
    EXPECT_TRUE(foundScanSource) << "ScanSourceOp should be created from BaseTableOp";
    EXPECT_TRUE(foundCreateDS) << "CreateDSOp should be created for result building";
    EXPECT_TRUE(foundForOp) << "ForOp should be created for iteration";
    EXPECT_TRUE(foundFinalizeOp) << "FinalizeOp should be created for result completion";
    EXPECT_EQ(atOpCount, 3) << "Should have 3 AtOps for 3 columns";
    EXPECT_EQ(dsAppendCount, 1) << "Should have 1 DSAppendOp for column values";
    EXPECT_EQ(nextRowCount, 1) << "Should have 1 NextRowOp for row finalization";
    
    PGX_DEBUG("MaterializeOp to DSA pattern lowering test completed successfully");
}

TEST_F(RelAlgToDSALoweringTest, ReturnOpToYieldOpLowering) {
    PGX_DEBUG("Testing ReturnOp to YieldOp lowering");
    
    // Create a module with BaseTableOp containing ReturnOp
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_return_lowering", funcType);
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
    
    // Verify ReturnOp exists before lowering
    bool foundReturnBefore = false;
    func.walk([&](::pgx::mlir::relalg::ReturnOp op) {
        foundReturnBefore = true;
    });
    EXPECT_TRUE(foundReturnBefore) << "ReturnOp should exist before lowering";
    
    // Run the lowering pass
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Lowering pass should succeed";
    
    // Verify ReturnOp was converted to YieldOp
    bool foundReturnAfter = false;
    bool foundYieldAfter = false;
    
    func.walk([&](::pgx::mlir::relalg::ReturnOp op) {
        foundReturnAfter = true;
    });
    
    func.walk([&](::pgx::mlir::dsa::YieldOp op) {
        foundYieldAfter = true;
    });
    
    EXPECT_FALSE(foundReturnAfter) << "ReturnOp should be removed after lowering";
    EXPECT_TRUE(foundYieldAfter) << "YieldOp should be created after lowering";
    
    PGX_DEBUG("ReturnOp to YieldOp lowering test completed successfully");
}

TEST_F(RelAlgToDSALoweringTest, FullPipelineSelectStarFromTest) {
    PGX_DEBUG("Testing complete pipeline for SELECT * FROM test lowering");
    
    // Create a complete RelAlg IR representing SELECT * FROM test
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "select_star_from_test", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp for 'test' table
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(16384)); // Typical PostgreSQL test table OID
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    // Create MaterializeOp for SELECT * (all columns)
    builder.setInsertionPointAfter(baseTableOp);
    SmallVector<Attribute> allColumnAttrs;
    allColumnAttrs.push_back(builder.getStringAttr("id"));
    allColumnAttrs.push_back(builder.getStringAttr("name"));
    allColumnAttrs.push_back(builder.getStringAttr("value"));
    auto columnsArrayAttr = builder.getArrayAttr(allColumnAttrs);
    
    auto materializeOp = builder.create<::pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Count RelAlg operations before lowering
    int relalgOpsBefore = 0;
    func.walk([&](Operation* op) {
        if (llvm::isa<::pgx::mlir::relalg::BaseTableOp>(op) ||
            llvm::isa<::pgx::mlir::relalg::MaterializeOp>(op) ||
            llvm::isa<::pgx::mlir::relalg::ReturnOp>(op)) {
            relalgOpsBefore++;
        }
    });
    EXPECT_GT(relalgOpsBefore, 0) << "Should have RelAlg operations before lowering";
    
    // Run the lowering pass
    ASSERT_TRUE(succeeded(runLoweringPass(module))) << "Full pipeline lowering should succeed";
    
    // Verify complete DSA structure was generated
    bool foundScanSource = false;
    bool foundCreateDS = false;
    bool foundForOp = false;
    bool foundFinalizeOp = false;
    int totalAtOps = 0;
    int totalYieldOps = 0;
    std::string scanSourceJSON;
    
    func.walk([&](Operation* op) {
        if (auto scanOp = llvm::dyn_cast<::pgx::mlir::dsa::ScanSourceOp>(op)) {
            foundScanSource = true;
            scanSourceJSON = scanOp.getTableDescription().str();
        } else if (llvm::isa<::pgx::mlir::dsa::CreateDSOp>(op)) {
            foundCreateDS = true;
        } else if (llvm::isa<::pgx::mlir::dsa::ForOp>(op)) {
            foundForOp = true;
        } else if (llvm::isa<::pgx::mlir::dsa::FinalizeOp>(op)) {
            foundFinalizeOp = true;
        } else if (llvm::isa<::pgx::mlir::dsa::AtOp>(op)) {
            totalAtOps++;
        } else if (llvm::isa<::pgx::mlir::dsa::YieldOp>(op)) {
            totalYieldOps++;
        }
    });
    
    // Verify the complete DSA pipeline structure
    EXPECT_TRUE(foundScanSource) << "Should have ScanSourceOp for table scan";
    EXPECT_TRUE(foundCreateDS) << "Should have CreateDSOp for result builder";
    EXPECT_TRUE(foundForOp) << "Should have ForOp for tuple iteration";
    EXPECT_TRUE(foundFinalizeOp) << "Should have FinalizeOp for result completion";
    EXPECT_EQ(totalAtOps, 3) << "Should have 3 AtOps for 3 columns (id, name, value)";
    EXPECT_GE(totalYieldOps, 1) << "Should have at least 1 YieldOp for termination";
    
    // Verify scan source contains correct table information
    EXPECT_NE(scanSourceJSON.find("\"table\":\"test\""), std::string::npos)
        << "JSON should contain table name 'test': " << scanSourceJSON;
    EXPECT_NE(scanSourceJSON.find("\"oid\":16384"), std::string::npos)
        << "JSON should contain table OID: " << scanSourceJSON;
    
    // Verify no RelAlg operations remain after complete lowering
    int relalgOpsAfter = 0;
    func.walk([&](Operation* op) {
        if (llvm::isa<::pgx::mlir::relalg::BaseTableOp>(op) ||
            llvm::isa<::pgx::mlir::relalg::MaterializeOp>(op) ||
            llvm::isa<::pgx::mlir::relalg::ReturnOp>(op)) {
            relalgOpsAfter++;
        }
    });
    EXPECT_EQ(relalgOpsAfter, 0) << "No RelAlg operations should remain after complete lowering";
    
    PGX_DEBUG("Full pipeline SELECT * FROM test lowering completed successfully");
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
    
    func.walk([&](::pgx::mlir::dsa::ForOp forOp) {
        // Verify ForOp has proper block structure
        EXPECT_EQ(forOp.getBody().getBlocks().size(), 1)
            << "ForOp should have exactly one block";
        
        Block &forBlock = forOp.getBody().front();
        EXPECT_EQ(forBlock.getNumArguments(), 1)
            << "ForOp block should have exactly one argument";
        
        auto argType = forBlock.getArgument(0).getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::RecordType>(argType))
            << "ForOp block argument should be RecordType";
    });
    
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