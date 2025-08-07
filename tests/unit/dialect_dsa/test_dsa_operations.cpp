#include "gtest/gtest.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "execution/logging.h"

using namespace mlir;

class DSAOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }
    
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test DSA types can be created
TEST_F(DSAOperationsTest, TestDSATypeCreation) {
    MLIR_PGX_DEBUG("UnitTest", "Testing DSA type creation");
    
    // Test RecordBatch type
    auto tupleType = builder->getTupleType({builder->getI32Type(), builder->getF64Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    
    EXPECT_TRUE(recordBatchType != nullptr);
    EXPECT_EQ(recordBatchType.getRowType(), tupleType);
    
    // Test Record type  
    auto recordType = ::pgx::mlir::dsa::RecordType::get(&context, tupleType);
    EXPECT_TRUE(recordType != nullptr);
    EXPECT_EQ(recordType.getRowType(), tupleType);
    
    // Test TableBuilder type
    auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    EXPECT_TRUE(tableBuilderType != nullptr);
    EXPECT_EQ(tableBuilderType.getRowType(), tupleType);
    
    // Test GenericIterable type
    auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "test_iterator");
    EXPECT_TRUE(iterableType != nullptr);
    
    MLIR_PGX_DEBUG("UnitTest", "DSA type creation test passed");
}

// Test ScanSource operation
TEST_F(DSAOperationsTest, TestScanSourceOp) {
    MLIR_PGX_DEBUG("UnitTest", "Testing ScanSourceOp");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create types
    auto tupleType = builder->getTupleType({builder->getI32Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "table_iterator");
    
    // Create ScanSourceOp
    auto scanSourceOp = builder->create<::pgx::mlir::dsa::ScanSourceOp>(
        builder->getUnknownLoc(), 
        iterableType,
        builder->getStringAttr("{\"table_oid\":12345}"));
    
    EXPECT_TRUE(scanSourceOp != nullptr);
    EXPECT_EQ(scanSourceOp.getResult().getType(), iterableType);
    EXPECT_EQ(scanSourceOp.getTableDescription().str(), "{\"table_oid\":12345}");
    
    MLIR_PGX_DEBUG("UnitTest", "ScanSourceOp test passed");
}

// Test CreateDS and data structure building operations
TEST_F(DSAOperationsTest, TestDataStructureBuilding) {
    MLIR_PGX_DEBUG("UnitTest", "Testing data structure building operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create types
    auto tupleType = builder->getTupleType({builder->getI32Type(), builder->getF64Type()});
    auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    // Test CreateDSOp
    auto createDSOp = builder->create<::pgx::mlir::dsa::CreateDSOp>(
        builder->getUnknownLoc(), tableBuilderType);
    
    EXPECT_TRUE(createDSOp != nullptr);
    EXPECT_EQ(createDSOp.getResult().getType(), tableBuilderType);
    
    // Test DSAppendOp with multiple values
    auto intValue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 42, builder->getI32Type());
    auto floatValue = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getF64FloatAttr(3.14));
    
    auto dsAppendOp = builder->create<::pgx::mlir::dsa::DSAppendOp>(
        builder->getUnknownLoc(), 
        createDSOp.getResult(), 
        ValueRange{intValue.getResult(), floatValue.getResult()});
    
    EXPECT_TRUE(dsAppendOp != nullptr);
    
    // Test NextRowOp
    auto nextRowOp = builder->create<::pgx::mlir::dsa::NextRowOp>(
        builder->getUnknownLoc(), createDSOp.getResult());
    
    EXPECT_TRUE(nextRowOp != nullptr);
    
    // Test FinalizeOp
    auto tableType = ::pgx::mlir::dsa::TableType::get(&context);
    auto finalizeOp = builder->create<::pgx::mlir::dsa::FinalizeOp>(
        builder->getUnknownLoc(), tableType, createDSOp.getResult());
    
    EXPECT_TRUE(finalizeOp != nullptr);
    EXPECT_EQ(finalizeOp.getResult().getType(), tableType);
    
    MLIR_PGX_DEBUG("UnitTest", "Data structure building operations test passed");
}

// Test AtOp for field extraction (using current string-based API)
TEST_F(DSAOperationsTest, TestAtOpFieldExtraction) {
    MLIR_PGX_DEBUG("UnitTest", "Testing AtOp for field extraction");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create record type
    auto tupleType = builder->getTupleType({builder->getI32Type(), builder->getF64Type()});
    auto recordType = ::pgx::mlir::dsa::RecordType::get(&context, tupleType);
    
    // Create a mock record value (in real code this would come from ForOp)
    auto funcType = builder->getFunctionType({recordType}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_at", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    Value recordArg = entryBlock->getArgument(0);
    
    // Test AtOp for column extraction
    auto atOp = builder->create<::pgx::mlir::dsa::AtOp>(
        builder->getUnknownLoc(), 
        builder->getI32Type(),
        recordArg, 
        builder->getStringAttr("column1"));
    
    EXPECT_TRUE(atOp != nullptr);
    EXPECT_EQ(atOp.getResult().getType(), builder->getI32Type());
    EXPECT_EQ(atOp.getColumnName(), "column1");
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "AtOp field extraction test passed");
}

// Test ForOp creation and verification
TEST_F(DSAOperationsTest, TestForOpCreation) {
    MLIR_PGX_DEBUG("UnitTest", "Testing ForOp creation");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_for", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create input iterable
    auto tupleType = builder->getTupleType({builder->getI32Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "table_iterator");
    
    auto scanSourceOp = builder->create<::pgx::mlir::dsa::ScanSourceOp>(
        builder->getUnknownLoc(), 
        iterableType,
        builder->getStringAttr("{\"table_oid\":12345}"));
    
    // Create ForOp with correct API
    auto forOp = builder->create<::pgx::mlir::dsa::ForOp>(
        builder->getUnknownLoc(), 
        TypeRange{}, 
        scanSourceOp.getResult());
    
    // Create body for ForOp
    Region& forRegion = forOp.getBody();
    Block* forBlock = &forRegion.emplaceBlock();
    forBlock->addArgument(recordBatchType, builder->getUnknownLoc());
    builder->setInsertionPointToStart(forBlock);
    
    // Add YieldOp to complete the ForOp body
    builder->create<::pgx::mlir::dsa::YieldOp>(builder->getUnknownLoc(), ValueRange{});
    
    builder->setInsertionPointAfter(forOp);
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(forOp.verify().succeeded());
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "ForOp creation test passed");
}

// Test Flag operations for control flow
TEST_F(DSAOperationsTest, TestFlagOperations) {
    MLIR_PGX_DEBUG("UnitTest", "Testing DSA flag operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create function
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_flags", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Test CreateFlagOp
    auto flagType = ::pgx::mlir::dsa::FlagType::get(&context);
    auto createFlagOp = builder->create<::pgx::mlir::dsa::CreateFlagOp>(
        builder->getUnknownLoc(), flagType);
    
    EXPECT_TRUE(createFlagOp != nullptr);
    EXPECT_EQ(createFlagOp.getFlag().getType(), flagType);
    
    // Test SetFlagOp
    auto constTrue = builder->create<arith::ConstantIntOp>(
        builder->getUnknownLoc(), 1, builder->getI1Type());
    auto setFlagOp = builder->create<::pgx::mlir::dsa::SetFlagOp>(
        builder->getUnknownLoc(), 
        createFlagOp.getFlag(), 
        constTrue.getResult());
    
    EXPECT_TRUE(setFlagOp != nullptr);
    
    // Test GetFlagOp
    auto getFlagOp = builder->create<::pgx::mlir::dsa::GetFlagOp>(
        builder->getUnknownLoc(), 
        builder->getI1Type(), 
        createFlagOp.getFlag());
    
    EXPECT_TRUE(getFlagOp != nullptr);
    EXPECT_EQ(getFlagOp.getRes().getType(), builder->getI1Type());
    
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "Flag operations test passed");
}