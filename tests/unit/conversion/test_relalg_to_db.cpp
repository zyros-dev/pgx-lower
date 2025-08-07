#include "gtest/gtest.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "execution/logging.h"

using namespace mlir;

class RelAlgToDBConversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }
    
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test BaseTableOp to GetExternalOp conversion
TEST_F(RelAlgToDBConversionTest, TestBaseTableToGetExternalConversion) {
    MLIR_PGX_DEBUG("UnitTest", "Testing BaseTableOp to GetExternalOp conversion");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto relationAttr = ::pgx::mlir::relalg::RelationAttr::get(&context, "test_table");
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        ::pgx::mlir::relalg::BagType::get(&context),
        relationAttr,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345));  // Table OID
    
    EXPECT_TRUE(baseTableOp != nullptr);
    
    // Set up conversion
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::db::DBDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
    
    RewritePatternSet patterns(&context);
    patterns.add<::pgx_conversion::BaseTableToExternalSourcePattern>(&context);
    
    // Apply conversion
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(applyPartialConversion(funcOp, target, std::move(patterns)).succeeded());
    
    // Verify conversion result - should have GetExternalOp
    bool foundGetExternalOp = false;
    funcOp.walk([&](::pgx::db::GetExternalOp op) {
        foundGetExternalOp = true;
        EXPECT_EQ(op.getTableOid(), 12345);
    });
    
    EXPECT_TRUE(foundGetExternalOp);
    MLIR_PGX_DEBUG("UnitTest", "BaseTableOp conversion test passed");
}

// Test GetColumnOp to GetFieldOp conversion  
TEST_F(RelAlgToDBConversionTest, TestGetColumnToGetFieldConversion) {
    MLIR_PGX_DEBUG("UnitTest", "Testing GetColumnOp to GetFieldOp conversion");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_getcolumn_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a mock external source (would be result of BaseTable conversion)
    auto externalSourceType = ::pgx::db::ExternalSourceType::get(&context);
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(), externalSourceType, builder->getI64IntegerAttr(12345));
    
    // Create GetColumnOp
    auto columnRefAttr = ::pgx::mlir::relalg::ColumnRefAttr::get(
        &context, 
        builder->getStringAttr("test_column"),
        builder->getI32IntegerAttr(0));
    
    auto getColumnOp = builder->create<::pgx::mlir::relalg::GetColumnOp>(
        builder->getUnknownLoc(),
        builder->getI32Type(),
        getExternalOp.getResult(),
        columnRefAttr);
    
    EXPECT_TRUE(getColumnOp != nullptr);
    
    // Set up conversion
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::db::DBDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<::pgx::mlir::relalg::GetColumnOp>();
    
    RewritePatternSet patterns(&context);
    patterns.add<::pgx_conversion::GetColumnToGetFieldPattern>(&context);
    
    // Apply conversion
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(applyPartialConversion(funcOp, target, std::move(patterns)).succeeded());
    
    // Verify conversion result - should have GetFieldOp
    bool foundGetFieldOp = false;
    funcOp.walk([&](::pgx::db::GetFieldOp op) {
        foundGetFieldOp = true;
        EXPECT_TRUE(op.getHandle().getType().isa<::pgx::db::ExternalSourceType>());
    });
    
    EXPECT_TRUE(foundGetFieldOp);
    MLIR_PGX_DEBUG("UnitTest", "GetColumnOp conversion test passed");
}

// Test MaterializeOp to StreamResultsOp conversion
TEST_F(RelAlgToDBConversionTest, TestMaterializeToStreamResultsConversion) {
    MLIR_PGX_DEBUG("UnitTest", "Testing MaterializeOp to StreamResultsOp conversion");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_materialize_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create mock input (would be result of other RelAlg operations)
    auto bagType = ::pgx::mlir::relalg::BagType::get(&context);
    auto mockInputOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(), 
        ::pgx::db::ExternalSourceType::get(&context),
        builder->getI64IntegerAttr(12345));
    
    // Create MaterializeOp
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(),
        bagType,
        mockInputOp.getResult());
    
    EXPECT_TRUE(materializeOp != nullptr);
    
    // Set up conversion
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::db::DBDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<::pgx::mlir::relalg::MaterializeOp>();
    
    RewritePatternSet patterns(&context);
    patterns.add<::pgx_conversion::MaterializeToStreamResultsPattern>(&context);
    
    // Apply conversion
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(applyPartialConversion(funcOp, target, std::move(patterns)).succeeded());
    
    // Verify conversion result - should have StreamResultsOp and SCF loop
    bool foundStreamResultsOp = false;
    bool foundSCFForOp = false;
    bool foundStoreResultOp = false;
    
    funcOp.walk([&](Operation* op) {
        if (auto streamOp = dyn_cast<::pgx::db::StreamResultsOp>(op)) {
            foundStreamResultsOp = true;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            foundSCFForOp = true;
        }
        if (auto storeOp = dyn_cast<::pgx::db::StoreResultOp>(op)) {
            foundStoreResultOp = true;
        }
    });
    
    EXPECT_TRUE(foundStreamResultsOp);
    EXPECT_TRUE(foundSCFForOp);
    EXPECT_TRUE(foundStoreResultOp);
    
    MLIR_PGX_DEBUG("UnitTest", "MaterializeOp conversion test passed");
}

// Test full pass execution
TEST_F(RelAlgToDBConversionTest, TestFullPassExecution) {
    MLIR_PGX_DEBUG("UnitTest", "Testing full RelAlg to DB conversion pass");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function with RelAlg operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_full_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create RelAlg operations
    auto relationAttr = ::pgx::mlir::relalg::RelationAttr::get(&context, "test_table");
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        ::pgx::mlir::relalg::BagType::get(&context),
        relationAttr,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345));
    
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(),
        ::pgx::mlir::relalg::BagType::get(&context),
        baseTableOp.getResult());
    
    auto returnOp = builder->create<::pgx::mlir::relalg::ReturnOp>(
        builder->getUnknownLoc(), materializeOp.getResult());
    
    // Apply the full pass
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(::pgx_conversion::createRelAlgToDBPass());
    
    EXPECT_TRUE(pm.run(module).succeeded());
    
    // Verify all RelAlg operations are gone and replaced with DB operations
    bool hasRelAlgOps = false;
    bool hasDBOps = false;
    
    funcOp.walk([&](Operation* op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == ::pgx::mlir::relalg::RelAlgDialect::getDialectNamespace()) {
            hasRelAlgOps = true;
        }
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == ::pgx::db::DBDialect::getDialectNamespace()) {
            hasDBOps = true;
        }
    });
    
    EXPECT_FALSE(hasRelAlgOps);  // All RelAlg ops should be converted
    EXPECT_TRUE(hasDBOps);       // Should have DB ops
    
    MLIR_PGX_DEBUG("UnitTest", "Full pass execution test passed");
}