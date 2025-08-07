#include "gtest/gtest.h"
#include "mlir/Conversion/DBToDSA/DBToDSA.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "execution/logging.h"

using namespace mlir;

class DBToDSAConversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }
    
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test GetExternalOp to ScanSourceOp conversion
TEST_F(DBToDSAConversionTest, TestGetExternalToScanSourceConversion) {
    MLIR_PGX_DEBUG("UnitTest", "Testing GetExternalOp to ScanSourceOp conversion");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_getexternal_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create GetExternalOp
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        builder->getI64IntegerAttr(12345));  // Table OID
    
    EXPECT_TRUE(getExternalOp != nullptr);
    
    // Set up conversion
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<::pgx::db::GetExternalOp>();
    
    RewritePatternSet patterns(&context);
    patterns.add<::pgx_conversion::GetExternalToScanSourcePattern>(&context);
    
    // Apply conversion
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(applyPartialConversion(funcOp, target, std::move(patterns)).succeeded());
    
    // Verify conversion result - should have ScanSourceOp
    bool foundScanSourceOp = false;
    funcOp.walk([&](::pgx::mlir::dsa::ScanSourceOp op) {
        foundScanSourceOp = true;
        
        // Verify JSON descriptor contains table OID
        auto jsonDesc = op.getTableDescription().str();
        EXPECT_TRUE(jsonDesc.find("12345") != std::string::npos);
        
        // Verify return type is GenericIterable
        EXPECT_TRUE(op.getResult().getType().isa<::pgx::mlir::dsa::GenericIterableType>());
    });
    
    EXPECT_TRUE(foundScanSourceOp);
    MLIR_PGX_DEBUG("UnitTest", "GetExternalOp conversion test passed");
}

// Test GetFieldOp to AtOp conversion
TEST_F(DBToDSAConversionTest, TestGetFieldToAtConversion) {
    MLIR_PGX_DEBUG("UnitTest", "Testing GetFieldOp to AtOp conversion");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_getfield_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create mock external source handle
    auto externalSourceOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        builder->getI64IntegerAttr(12345));
    
    // Create GetFieldOp
    auto getFieldOp = builder->create<::pgx::db::GetFieldOp>(
        builder->getUnknownLoc(),
        ::pgx::db::NullableI32Type::get(&context),
        externalSourceOp.getResult(),
        builder->getIndexAttr(2),     // Field index
        builder->getI32IntegerAttr(23));  // Type OID (INT4OID)
    
    EXPECT_TRUE(getFieldOp != nullptr);
    
    // Set up conversion
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<::pgx::db::DBDialect>();  // Keep other DB ops legal
    target.addIllegalOp<::pgx::db::GetFieldOp>();
    
    RewritePatternSet patterns(&context);
    patterns.add<::pgx_conversion::GetFieldToAtPattern>(&context);
    
    // Apply conversion
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(applyPartialConversion(funcOp, target, std::move(patterns)).succeeded());
    
    // Verify conversion result - should have AtOp
    bool foundAtOp = false;
    funcOp.walk([&](::pgx::mlir::dsa::AtOp op) {
        foundAtOp = true;
        
        // Verify field position matches
        auto posAttr = op.getPosAttr();
        EXPECT_EQ(posAttr.getInt(), 2);
        
        // Verify result type
        EXPECT_TRUE(op.getVal().getType().isa<IntegerType>());
    });
    
    EXPECT_TRUE(foundAtOp);
    MLIR_PGX_DEBUG("UnitTest", "GetFieldOp conversion test passed");
}

// Test StreamResultsOp to DSA finalization conversion
TEST_F(DBToDSAConversionTest, TestStreamResultsToFinalizeConversion) {
    MLIR_PGX_DEBUG("UnitTest", "Testing StreamResultsOp to DSA finalization conversion");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_streamresults_conversion", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create StreamResultsOp
    auto streamResultsOp = builder->create<::pgx::db::StreamResultsOp>(
        builder->getUnknownLoc());
    
    EXPECT_TRUE(streamResultsOp != nullptr);
    
    // Set up conversion
    ConversionTarget target(context);
    target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<::pgx::db::StreamResultsOp>();
    
    RewritePatternSet patterns(&context);
    patterns.add<::pgx_conversion::StreamResultsToFinalizePattern>(&context);
    
    // Apply conversion
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(applyPartialConversion(funcOp, target, std::move(patterns)).succeeded());
    
    // Verify conversion result - should have DSA table building operations
    bool foundCreateDSOp = false;
    bool foundDSAppendOp = false;
    bool foundNextRowOp = false;
    bool foundFinalizeOp = false;
    
    funcOp.walk([&](Operation* op) {
        if (dyn_cast<::pgx::mlir::dsa::CreateDSOp>(op)) {
            foundCreateDSOp = true;
        }
        if (dyn_cast<::pgx::mlir::dsa::DSAppendOp>(op)) {
            foundDSAppendOp = true;
        }
        if (dyn_cast<::pgx::mlir::dsa::NextRowOp>(op)) {
            foundNextRowOp = true;
        }
        if (dyn_cast<::pgx::mlir::dsa::FinalizeOp>(op)) {
            foundFinalizeOp = true;
        }
    });
    
    EXPECT_TRUE(foundCreateDSOp);
    EXPECT_TRUE(foundDSAppendOp);
    EXPECT_TRUE(foundNextRowOp);
    EXPECT_TRUE(foundFinalizeOp);
    
    MLIR_PGX_DEBUG("UnitTest", "StreamResultsOp conversion test passed");
}

// Test full DB to DSA pass execution
TEST_F(DBToDSAConversionTest, TestFullPassExecution) {
    MLIR_PGX_DEBUG("UnitTest", "Testing full DB to DSA conversion pass");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function with DB operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_full_db_to_dsa", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create DB operations
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        builder->getI64IntegerAttr(12345));
    
    auto getFieldOp = builder->create<::pgx::db::GetFieldOp>(
        builder->getUnknownLoc(),
        ::pgx::db::NullableI32Type::get(&context),
        getExternalOp.getResult(),
        builder->getIndexAttr(0),
        builder->getI32IntegerAttr(23));
    
    auto streamResultsOp = builder->create<::pgx::db::StreamResultsOp>(
        builder->getUnknownLoc());
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Apply the full pass
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(::pgx_conversion::createDBToDSAPass());
    
    EXPECT_TRUE(pm.run(module).succeeded());
    
    // Verify targeted DB operations are converted but others remain
    bool hasGetExternalOp = false;
    bool hasGetFieldOp = false;
    bool hasStreamResultsOp = false;
    bool hasDSAOps = false;
    
    funcOp.walk([&](Operation* op) {
        if (dyn_cast<::pgx::db::GetExternalOp>(op)) {
            hasGetExternalOp = true;
        }
        if (dyn_cast<::pgx::db::GetFieldOp>(op)) {
            hasGetFieldOp = true;
        }
        if (dyn_cast<::pgx::db::StreamResultsOp>(op)) {
            hasStreamResultsOp = true;
        }
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == ::pgx::mlir::dsa::DSADialect::getDialectNamespace()) {
            hasDSAOps = true;
        }
    });
    
    // Targeted DB operations should be converted
    EXPECT_FALSE(hasGetExternalOp);
    EXPECT_FALSE(hasGetFieldOp); 
    EXPECT_FALSE(hasStreamResultsOp);
    
    // Should have DSA operations
    EXPECT_TRUE(hasDSAOps);
    
    MLIR_PGX_DEBUG("UnitTest", "Full DB to DSA pass execution test passed");
}

// Test end-to-end pipeline: RelAlg → DB → DSA
TEST_F(DBToDSAConversionTest, TestEndToEndPipeline) {
    MLIR_PGX_DEBUG("UnitTest", "Testing end-to-end RelAlg → DB → DSA pipeline");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function with RelAlg operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_end_to_end_pipeline", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create RelAlg operations (starting point)
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
    
    // Apply full pipeline: RelAlg → DB → DSA
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(::pgx_conversion::createRelAlgToDBPass());
    pm.addNestedPass<func::FuncOp>(::pgx_conversion::createDBToDSAPass());
    
    EXPECT_TRUE(pm.run(module).succeeded());
    
    // Verify pipeline result - should have DSA operations and no RelAlg operations
    bool hasRelAlgOps = false;
    bool hasTargetedDBOps = false;  // GetExternal, GetField, StreamResults should be gone
    bool hasDSAOps = false;
    
    funcOp.walk([&](Operation* op) {
        if (op->getDialect()) {
            auto dialectNamespace = op->getDialect()->getNamespace();
            
            if (dialectNamespace == ::pgx::mlir::relalg::RelAlgDialect::getDialectNamespace()) {
                hasRelAlgOps = true;
            }
            
            if (dyn_cast<::pgx::db::GetExternalOp>(op) ||
                dyn_cast<::pgx::db::GetFieldOp>(op) ||
                dyn_cast<::pgx::db::StreamResultsOp>(op)) {
                hasTargetedDBOps = true;
            }
            
            if (dialectNamespace == ::pgx::mlir::dsa::DSADialect::getDialectNamespace()) {
                hasDSAOps = true;
            }
        }
    });
    
    // Final result should be DSA operations with no RelAlg or targeted DB operations
    EXPECT_FALSE(hasRelAlgOps);       // All RelAlg should be converted
    EXPECT_FALSE(hasTargetedDBOps);   // Targeted DB ops should be converted to DSA
    EXPECT_TRUE(hasDSAOps);           // Should have DSA operations
    
    // Verify we have the core DSA operations for the complete pipeline
    bool hasScanSourceOp = false;
    bool hasFinalizeOp = false;
    
    funcOp.walk([&](Operation* op) {
        if (dyn_cast<::pgx::mlir::dsa::ScanSourceOp>(op)) {
            hasScanSourceOp = true;
        }
        if (dyn_cast<::pgx::mlir::dsa::FinalizeOp>(op)) {
            hasFinalizeOp = true;
        }
    });
    
    EXPECT_TRUE(hasScanSourceOp);  // From BaseTable conversion
    EXPECT_TRUE(hasFinalizeOp);    // From Materialize conversion
    
    MLIR_PGX_DEBUG("UnitTest", "End-to-end pipeline test passed");
}