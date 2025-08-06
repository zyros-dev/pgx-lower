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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class WorkingLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        
        PGX_DEBUG("WorkingLoweringTest setup completed");
    }

    // Helper function to run a simple BaseTableOp lowering
    LogicalResult runBaseTableLowering(ModuleOp module) {
        ConversionTarget target(context);
        target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        
        RewritePatternSet patterns(&context);
        patterns.add<mlir::pgx_conversion::BaseTableToScanSourcePattern>(&context);
        
        return applyPartialConversion(module, target, std::move(patterns));
    }

    MLIRContext context;
    OpBuilder builder{&context};
};

TEST_F(WorkingLoweringTest, SimpleBaseTableOpLowering) {
    PGX_DEBUG("Testing simple BaseTableOp lowering that works");
    
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_simple", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("customers"),
        builder.getI64IntegerAttr(12345));
    
    // Initialize region
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Count RelAlg ops before
    int relalgOpsBefore = 0;
    func.walk([&](::pgx::mlir::relalg::BaseTableOp) { relalgOpsBefore++; });
    EXPECT_EQ(relalgOpsBefore, 1) << "Should have 1 BaseTableOp before lowering";
    
    // Run lowering
    ASSERT_TRUE(succeeded(runBaseTableLowering(module))) << "BaseTable lowering should succeed";
    
    // Verify conversion
    int relalgOpsAfter = 0;
    int dsaOpsAfter = 0;
    std::string scanJSON;
    
    func.walk([&](::pgx::mlir::relalg::BaseTableOp) { relalgOpsAfter++; });
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp scanOp) { 
        dsaOpsAfter++; 
        scanJSON = scanOp.getTableDescription().str();
    });
    
    EXPECT_EQ(relalgOpsAfter, 0) << "BaseTableOp should be removed";
    EXPECT_EQ(dsaOpsAfter, 1) << "ScanSourceOp should be created";
    EXPECT_NE(scanJSON.find("\"table\":\"customers\""), std::string::npos) << "JSON should contain table name";
    EXPECT_NE(scanJSON.find("\"oid\":12345"), std::string::npos) << "JSON should contain OID";
    
    PGX_DEBUG("Simple BaseTableOp lowering completed successfully");
}

TEST_F(WorkingLoweringTest, VerifyBaseTableOpConversion) {
    PGX_DEBUG("Testing BaseTableOp conversion details");
    
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_conversion_details", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp with specific attributes
    const std::string tableName = "orders";
    const int64_t tableOid = 67890;
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr(tableName),
        builder.getI64IntegerAttr(tableOid));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run lowering
    ASSERT_TRUE(succeeded(runBaseTableLowering(module))) << "Lowering should succeed";
    
    // Detailed verification
    bool foundScanSource = false;
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp scanOp) {
        foundScanSource = true;
        
        // Verify type
        auto resultType = scanOp.getResult().getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::GenericIterableType>(resultType))
            << "ScanSourceOp should return GenericIterableType";
        
        // Verify JSON content
        std::string json = scanOp.getTableDescription().str();
        EXPECT_NE(json.find("\"table\":\"" + tableName + "\""), std::string::npos)
            << "JSON should contain correct table name: " << json;
        EXPECT_NE(json.find("\"oid\":" + std::to_string(tableOid)), std::string::npos)
            << "JSON should contain correct OID: " << json;
        
        PGX_DEBUG("ScanSourceOp JSON: " + json);
    });
    
    EXPECT_TRUE(foundScanSource) << "Should find exactly one ScanSourceOp";
    
    // Verify no RelAlg operations remain
    func.walk([&](::pgx::mlir::relalg::BaseTableOp) {
        FAIL() << "BaseTableOp should not remain after lowering";
    });
    
    PGX_DEBUG("BaseTableOp conversion verification completed successfully");
}

TEST_F(WorkingLoweringTest, MultipleBaseTableOpsLowering) {
    PGX_DEBUG("Testing multiple BaseTableOp lowering");
    
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_multiple", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create first BaseTableOp
    auto baseTable1 = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("table1"),
        builder.getI64IntegerAttr(111));
    
    Block *body1 = &baseTable1.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard1(builder);
    builder.setInsertionPointToStart(body1);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    // Create second BaseTableOp
    builder.setInsertionPointAfter(baseTable1);
    auto baseTable2 = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr("table2"),
        builder.getI64IntegerAttr(222));
    
    Block *body2 = &baseTable2.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard2(builder);
    builder.setInsertionPointToStart(body2);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTable2);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run lowering
    ASSERT_TRUE(succeeded(runBaseTableLowering(module))) << "Multiple BaseTable lowering should succeed";
    
    // Verify both were converted
    int scanSourceCount = 0;
    std::vector<std::string> jsonDescriptions;
    
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp scanOp) {
        scanSourceCount++;
        jsonDescriptions.push_back(scanOp.getTableDescription().str());
    });
    
    EXPECT_EQ(scanSourceCount, 2) << "Should have 2 ScanSourceOps after lowering";
    
    // Verify both tables are represented
    bool foundTable1 = false, foundTable2 = false;
    for (const auto& json : jsonDescriptions) {
        if (json.find("\"table\":\"table1\"") != std::string::npos && 
            json.find("\"oid\":111") != std::string::npos) {
            foundTable1 = true;
        }
        if (json.find("\"table\":\"table2\"") != std::string::npos &&
            json.find("\"oid\":222") != std::string::npos) {
            foundTable2 = true;
        }
    }
    
    EXPECT_TRUE(foundTable1) << "Should find table1 in scan operations";
    EXPECT_TRUE(foundTable2) << "Should find table2 in scan operations";
    
    // Verify no RelAlg ops remain
    int remainingBaseTableOps = 0;
    func.walk([&](::pgx::mlir::relalg::BaseTableOp) { remainingBaseTableOps++; });
    EXPECT_EQ(remainingBaseTableOps, 0) << "No BaseTableOps should remain";
    
    PGX_DEBUG("Multiple BaseTableOps lowering completed successfully");
}

TEST_F(WorkingLoweringTest, BaseTableOpTypePreservation) {
    PGX_DEBUG("Testing that BaseTableOp lowering preserves type correctness");
    
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_type_preservation", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create BaseTableOp
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("type_test"),
        builder.getI64IntegerAttr(999));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify initial type
    EXPECT_TRUE(llvm::isa<::pgx::mlir::relalg::TupleStreamType>(baseTableOp.getResult().getType()))
        << "BaseTableOp should initially return TupleStreamType";
    
    // Run lowering
    ASSERT_TRUE(succeeded(runBaseTableLowering(module))) << "Type preservation lowering should succeed";
    
    // Verify type transformation
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp scanOp) {
        auto resultType = scanOp.getResult().getType();
        EXPECT_TRUE(llvm::isa<::pgx::mlir::dsa::GenericIterableType>(resultType))
            << "ScanSourceOp should return GenericIterableType";
        
        // Verify the type is correctly instantiated
        auto genericIterableType = llvm::cast<::pgx::mlir::dsa::GenericIterableType>(resultType);
        EXPECT_TRUE(genericIterableType) << "GenericIterableType should be valid";
    });
    
    PGX_DEBUG("BaseTableOp type preservation test completed successfully");
}

TEST_F(WorkingLoweringTest, BaseTableOpAttributeHandling) {
    PGX_DEBUG("Testing BaseTableOp attribute handling in lowering");
    
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_attributes", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Test with extreme values
    const std::string longTableName = "very_long_table_name_with_special_chars_123_$%^&*()";
    const int64_t largeOid = 9223372036854775807LL; // max int64_t
    
    auto baseTableOp = builder.create<::pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        ::pgx::mlir::relalg::TupleStreamType::get(&context),
        builder.getStringAttr(longTableName),
        builder.getI64IntegerAttr(largeOid));
    
    Block *baseTableBody = &baseTableOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(baseTableBody);
    builder.create<::pgx::mlir::relalg::ReturnOp>(builder.getUnknownLoc());
    
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run lowering
    ASSERT_TRUE(succeeded(runBaseTableLowering(module))) << "Attribute handling lowering should succeed";
    
    // Verify attribute preservation
    func.walk([&](::pgx::mlir::dsa::ScanSourceOp scanOp) {
        std::string json = scanOp.getTableDescription().str();
        
        // Verify table name preserved
        EXPECT_NE(json.find("\"table\":\"" + longTableName + "\""), std::string::npos)
            << "Long table name should be preserved: " << json;
        
        // Verify OID preserved  
        EXPECT_NE(json.find("\"oid\":" + std::to_string(largeOid)), std::string::npos)
            << "Large OID should be preserved: " << json;
    });
    
    PGX_DEBUG("BaseTableOp attribute handling test completed successfully");
}