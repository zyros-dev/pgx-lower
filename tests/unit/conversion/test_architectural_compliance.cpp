#include "gtest/gtest.h"
#include "execution/logging.h"

#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBOps.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class ArchitecturalComplianceTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        builder = std::make_unique<OpBuilder>(&context);
        
        PGX_DEBUG("Architectural compliance test setup completed");
    }
    
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test that dialects can be loaded without issues  
TEST_F(ArchitecturalComplianceTest, TestDialectLoading) {
    PGX_INFO("Testing dialect loading for architectural compliance");
    
    // Verify all three dialects are loaded
    auto* relalgDialect = context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
    auto* dbDialect = context.getOrLoadDialect<::pgx::db::DBDialect>();
    auto* dsaDialect = context.getOrLoadDialect<::mlir::dsa::DSADialect>();
    
    EXPECT_NE(relalgDialect, nullptr);
    EXPECT_NE(dbDialect, nullptr);
    EXPECT_NE(dsaDialect, nullptr);
    
    PGX_INFO("✅ ARCHITECTURAL COMPLIANCE: All dialects loaded successfully");
}

// Test basic RelAlg operations can be created
TEST_F(ArchitecturalComplianceTest, TestRelAlgOperationCreation) {
    PGX_INFO("Testing RelAlg operation creation");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_relalg_ops", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Test BaseTableOp creation with current API
    auto tupleStreamType = ::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345));
    
    EXPECT_TRUE(baseTableOp != nullptr);
    EXPECT_TRUE(baseTableOp.getResult().getType() == tupleStreamType);
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify the function is well-formed
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    PGX_INFO("✅ ARCHITECTURAL COMPLIANCE: RelAlg operations created successfully");
}

// Test DB operations can be created
TEST_F(ArchitecturalComplianceTest, TestDBOperationCreation) {
    PGX_INFO("Testing DB operation creation");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_db_ops", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Test GetExternalOp creation with current API
    auto tableOidValue = builder->create<arith::ConstantIntOp>(builder->getUnknownLoc(), 12345, builder->getI64Type());
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        tableOidValue.getResult());
    
    EXPECT_TRUE(getExternalOp != nullptr);
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify the function is well-formed
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    PGX_INFO("✅ ARCHITECTURAL COMPLIANCE: DB operations created successfully");
}

// Test DSA operations can be created
TEST_F(ArchitecturalComplianceTest, TestDSAOperationCreation) {
    PGX_INFO("Testing DSA operation creation");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function to contain the operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_dsa_ops", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Test ScanSource creation with current API
    auto tupleType = builder->getTupleType({builder->getI32Type()});
    auto recordBatchType = ::mlir::dsa::RecordBatchType::get(&context, tupleType);
    auto iterableType = ::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "test_iterator");
    
    auto scanSourceOp = builder->create<::mlir::dsa::ScanSource>(
        builder->getUnknownLoc(), 
        iterableType,
        builder->getStringAttr("{\"table_oid\":12345}"));
    
    EXPECT_TRUE(scanSourceOp != nullptr);
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify the function is well-formed
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    PGX_INFO("✅ ARCHITECTURAL COMPLIANCE: DSA operations created successfully");
}

// Test that conversion passes exist and can be created
TEST_F(ArchitecturalComplianceTest, TestConversionPassesAvailable) {
    PGX_INFO("Testing conversion passes availability");
    
    // Test that conversion pass creation functions exist
    auto relAlgToDBPass = ::pgx_conversion::createRelAlgToDBPass();
    
    EXPECT_TRUE(relAlgToDBPass != nullptr);
    
    PGX_INFO("✅ ARCHITECTURAL COMPLIANCE: Conversion passes available");
}

// Test that PassManager can be configured with conversion passes
TEST_F(ArchitecturalComplianceTest, TestPassManagerConfiguration) {
    PGX_INFO("Testing PassManager configuration with conversion passes");
    
    PassManager pm(&context);
    
    // Add RelAlg to DB conversion pass
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // Add DB to DSA conversion pass  
    
    // Just test that the PassManager was configured successfully
    // We don't run it because we'd need valid MLIR module with RelAlg ops
    
    PGX_INFO("✅ ARCHITECTURAL COMPLIANCE: PassManager configured successfully");
}