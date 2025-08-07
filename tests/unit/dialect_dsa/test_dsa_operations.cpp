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

// Test new parameterized types match LingoDB patterns
TEST_F(DSAOperationsTest, TestParameterizedTypes) {
    MLIR_PGX_DEBUG("UnitTest", "Testing DSA parameterized types");
    
    // Test RecordBatch type with tuple parameter
    auto tupleType = builder->getTupleType({builder->getI32Type(), builder->getF64Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    
    EXPECT_TRUE(recordBatchType != nullptr);
    EXPECT_EQ(recordBatchType.getRowType(), tupleType);
    
    // Test Record type with tuple parameter  
    auto recordType = ::pgx::mlir::dsa::RecordType::get(&context, tupleType);
    EXPECT_TRUE(recordType != nullptr);
    EXPECT_EQ(recordType.getRowType(), tupleType);
    
    // Test TableBuilder type with tuple parameter
    auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    EXPECT_TRUE(tableBuilderType != nullptr);
    EXPECT_EQ(tableBuilderType.getRowType(), tupleType);
    
    MLIR_PGX_DEBUG("UnitTest", "Parameterized types test passed");
}

// Test new flag operations
TEST_F(DSAOperationsTest, TestFlagOperations) {
    MLIR_PGX_DEBUG("UnitTest", "Testing DSA flag operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Test CreateFlagOp
    auto flagType = ::pgx::mlir::dsa::FlagType::get(&context);
    auto createFlagOp = builder->create<::pgx::mlir::dsa::CreateFlagOp>(
        builder->getUnknownLoc(), flagType);
    
    EXPECT_TRUE(createFlagOp != nullptr);
    EXPECT_EQ(createFlagOp.getFlag().getType(), flagType);
    
    // Test SetFlagOp
    auto constTrue = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getBoolAttr(true));
    auto setFlagOp = builder->create<::pgx::mlir::dsa::SetFlagOp>(
        builder->getUnknownLoc(), createFlagOp.getFlag(), constTrue.getResult());
    
    EXPECT_TRUE(setFlagOp != nullptr);
    EXPECT_EQ(setFlagOp.getFlag(), createFlagOp.getFlag());
    
    // Test GetFlagOp
    auto getFlagOp = builder->create<::pgx::mlir::dsa::GetFlagOp>(
        builder->getUnknownLoc(), builder->getI1Type(), createFlagOp.getFlag());
    
    EXPECT_TRUE(getFlagOp != nullptr);
    EXPECT_EQ(getFlagOp.getRes().getType(), builder->getI1Type());
    
    MLIR_PGX_DEBUG("UnitTest", "Flag operations test passed");
}

// Test hashtable operations
TEST_F(DSAOperationsTest, TestHashtableOperations) {
    MLIR_PGX_DEBUG("UnitTest", "Testing DSA hashtable operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create hashtable type
    auto keyType = builder->getTupleType({builder->getI32Type()});
    auto valType = builder->getTupleType({builder->getF64Type()});
    auto hashtableType = ::pgx::mlir::dsa::JoinHashtableType::get(&context, keyType, valType);
    
    // Test hashtable creation
    auto createDSOp = builder->create<::pgx::mlir::dsa::CreateDSOp>(
        builder->getUnknownLoc(), hashtableType);
    
    EXPECT_TRUE(createDSOp != nullptr);
    
    // Test LookupOp  
    auto keyValue = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getI32IntegerAttr(42));
    
    auto genericIterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context,
        ::pgx::mlir::dsa::RecordType::get(&context, valType),
        "hashtable_iterator");
    
    auto lookupOp = builder->create<::pgx::mlir::dsa::LookupOp>(
        builder->getUnknownLoc(), genericIterableType,
        createDSOp.getResult(), keyValue.getResult());
    
    EXPECT_TRUE(lookupOp != nullptr);
    EXPECT_EQ(lookupOp.getIterable().getType(), genericIterableType);
    
    MLIR_PGX_DEBUG("UnitTest", "Hashtable operations test passed");
}

// Test nested ForOp pattern structure
TEST_F(DSAOperationsTest, TestNestedForOpPattern) {
    MLIR_PGX_DEBUG("UnitTest", "Testing nested ForOp pattern structure");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<func::FuncOp>(
        builder->getUnknownLoc(), "test_func", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create generic iterable input
    auto tupleType = builder->getTupleType({builder->getI32Type()});
    auto recordBatchType = ::pgx::mlir::dsa::RecordBatchType::get(&context, tupleType);
    auto iterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        &context, recordBatchType, "table_iterator");
    
    // Create scan source
    auto scanSourceOp = builder->create<::pgx::mlir::dsa::ScanSourceOp>(
        builder->getUnknownLoc(), iterableType, 
        builder->getStringAttr("{\"table\":\"test\"}"));
    
    // Create outer ForOp (record batch iteration)
    auto outerForOp = builder->create<::pgx::mlir::dsa::ForOp>(
        builder->getUnknownLoc(), TypeRange{}, scanSourceOp.getResult(), ValueRange{});
    
    Block* outerBlock = outerForOp.getBody();
    builder->setInsertionPointToStart(outerBlock);
    
    Value batchArg = outerBlock->getArgument(0);
    
    // Create inner ForOp (record iteration) 
    auto recordType = ::pgx::mlir::dsa::RecordType::get(&context, tupleType);
    auto innerForOp = builder->create<::pgx::mlir::dsa::ForOp>(
        builder->getUnknownLoc(), TypeRange{}, batchArg, ValueRange{});
    
    Block* innerBlock = innerForOp.getBody();
    builder->setInsertionPointToStart(innerBlock);
    
    Value recordArg = innerBlock->getArgument(0);
    
    // Test AtOp for field extraction
    auto atOp = builder->create<::pgx::mlir::dsa::AtOp>(
        builder->getUnknownLoc(), builder->getI32Type(),
        recordArg, builder->getI32IntegerAttr(0));
    
    EXPECT_TRUE(atOp != nullptr);
    EXPECT_EQ(atOp.getVal().getType(), builder->getI32Type());
    
    // Complete nested structure with yields
    builder->create<::pgx::mlir::dsa::YieldOp>(builder->getUnknownLoc(), ValueRange{});
    
    builder->setInsertionPointAfter(innerForOp);
    builder->create<::pgx::mlir::dsa::YieldOp>(builder->getUnknownLoc(), ValueRange{});
    
    builder->setInsertionPointAfter(outerForOp);
    builder->create<func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify the nested structure is well-formed
    EXPECT_TRUE(outerForOp.verify().succeeded());
    EXPECT_TRUE(innerForOp.verify().succeeded());
    EXPECT_TRUE(funcOp.verify().succeeded());
    
    MLIR_PGX_DEBUG("UnitTest", "Nested ForOp pattern test passed");
}

// Test result building operations
TEST_F(DSAOperationsTest, TestResultBuildingOperations) {
    MLIR_PGX_DEBUG("UnitTest", "Testing result building operations");
    
    auto module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
    
    // Create table builder
    auto tupleType = builder->getTupleType({builder->getI32Type(), builder->getF64Type()});
    auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
    
    auto createDSOp = builder->create<::pgx::mlir::dsa::CreateDSOp>(
        builder->getUnknownLoc(), tableBuilderType);
    
    EXPECT_TRUE(createDSOp != nullptr);
    
    // Test DSAppendOp with multiple values
    auto intValue = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getI32IntegerAttr(42));
    auto floatValue = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getF64FloatAttr(3.14));
    
    auto dsAppendOp = builder->create<::pgx::mlir::dsa::DSAppendOp>(
        builder->getUnknownLoc(), createDSOp.getResult(), 
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
    
    MLIR_PGX_DEBUG("UnitTest", "Result building operations test passed");
}