#include "gtest/gtest.h"
#include "execution/logging.h"

#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class ConversionBasicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }
    
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test basic RelAlg module creation
TEST_F(ConversionBasicsTest, TestRelAlgModuleCreation) {
    MLIR_PGX_DEBUG("UnitTest", "Testing RelAlg module creation");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a simple function with RelAlg BaseTable operation
    auto tableType = ::pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "query", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = ::pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        builder->getUnknownLoc(),
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345));
    
    // Create MaterializeOp
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder->getStringAttr("*"));
    auto columnsArrayAttr = builder->getArrayAttr(columnAttrs);
    
    auto materializeOp = builder->create<::pgx::mlir::relalg::MaterializeOp>(
        builder->getUnknownLoc(), 
        tableType, 
        baseTableOp.getResult(), 
        columnsArrayAttr);
    
    builder->create<func::ReturnOp>(
        builder->getUnknownLoc(), materializeOp.getResult());
    
    EXPECT_TRUE(funcOp.verify().succeeded());
    EXPECT_TRUE(module.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "RelAlg module creation test passed");
}

// Test that DB operations can be created independently
TEST_F(ConversionBasicsTest, TestDBOperationsIndependent) {
    MLIR_PGX_DEBUG("UnitTest", "Testing independent DB operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function with DB operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_db", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create GetExternalOp
    auto tableOidValue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 12345, builder->getI64Type());
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        tableOidValue.getResult());
    
    // Create GetFieldOp
    auto fieldIndexValue = builder->create<arith::ConstantIndexOp>(
        builder->getUnknownLoc(), 0);
    auto typeOidValue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 23, builder->getI32Type());
    auto getFieldOp = builder->create<::pgx::db::GetFieldOp>(
        builder->getUnknownLoc(),
        ::pgx::db::NullableI32Type::get(&context),
        getExternalOp.getResult(),
        fieldIndexValue.getResult(),
        typeOidValue.getResult());
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(funcOp.verify().succeeded());
    EXPECT_TRUE(module.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "Independent DB operations test passed");
}

// Test that DSA operations can be created independently  
TEST_F(ConversionBasicsTest, TestDSAOperationsIndependent) {
    MLIR_PGX_DEBUG("UnitTest", "Testing independent DSA operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function with DSA operations
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_dsa", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create ScanSourceOp
    auto tupleType = builder->getTupleType({builder->getI32Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "test_iterator");
    
    auto scanSourceOp = builder->create<::pgx::mlir::dsa::ScanSourceOp>(
        builder->getUnknownLoc(), 
        iterableType,
        builder->getStringAttr("{\"table_oid\":12345}"));
    
    // Create TableBuilder and operations
    auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    auto createDSOp = builder->create<::pgx::mlir::dsa::CreateDSOp>(
        builder->getUnknownLoc(), tableBuilderType);
    
    auto intValue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 42, builder->getI32Type());
    auto dsAppendOp = builder->create<::pgx::mlir::dsa::DSAppendOp>(
        builder->getUnknownLoc(), 
        createDSOp.getResult(), 
        ValueRange{intValue.getResult()});
    
    auto nextRowOp = builder->create<::pgx::mlir::dsa::NextRowOp>(
        builder->getUnknownLoc(), createDSOp.getResult());
    
    auto tableType = ::pgx::mlir::dsa::TableType::get(&context);
    auto finalizeOp = builder->create<::pgx::mlir::dsa::FinalizeOp>(
        builder->getUnknownLoc(), tableType, createDSOp.getResult());
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(funcOp.verify().succeeded());
    EXPECT_TRUE(module.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "Independent DSA operations test passed");
}

// Test conversion passes can be instantiated
TEST_F(ConversionBasicsTest, TestConversionPassInstantiation) {
    MLIR_PGX_DEBUG("UnitTest", "Testing conversion pass instantiation");
    
    // Test that we can create conversion passes
    auto relAlgToDBPass = ::pgx_conversion::createRelAlgToDBPass();
    
    EXPECT_TRUE(relAlgToDBPass != nullptr);
    
    // Test PassManager setup
    PassManager pm(&context);
    pm.addPass(::pgx_conversion::createRelAlgToDBPass());
    
    // PassManager should be configured without errors
    // We don't run conversion because we'd need proper input modules
    
    MLIR_PGX_DEBUG("UnitTest", "Conversion pass instantiation test passed");
}

// Test that conversions work without type conflicts
TEST_F(ConversionBasicsTest, TestConversionPrerequisites) {
    MLIR_PGX_DEBUG("UnitTest", "Testing conversion prerequisites");
    
    // Test that we can create individual operations that will be used in conversion
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "prerequisites", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create each operation type that participates in conversion
    auto tableOidValue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 12345, builder->getI64Type());
    
    auto getExternalOp = builder->create<::pgx::db::GetExternalOp>(
        builder->getUnknownLoc(),
        ::pgx::db::ExternalSourceType::get(&context),
        tableOidValue.getResult());
    
    // Create DSA ScanSourceOp
    auto tupleType = builder->getTupleType({builder->getI32Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "test_iterator");
    
    auto scanSourceOp = builder->create<::pgx::mlir::dsa::ScanSourceOp>(
        builder->getUnknownLoc(), 
        iterableType,
        builder->getStringAttr("{\"table_oid\":12345}"));
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(funcOp.verify().succeeded());
    EXPECT_TRUE(module.verify().succeeded());
    
    // Verify operations have expected attributes
    EXPECT_EQ(tableOidValue.value(), 12345);
    EXPECT_TRUE(getExternalOp.getTableOid() == tableOidValue.getResult());
    EXPECT_TRUE(scanSourceOp.getTableDescriptionAttr().getValue().contains("12345"));
    
    MLIR_PGX_DEBUG("UnitTest", "Conversion prerequisites test passed");
}


