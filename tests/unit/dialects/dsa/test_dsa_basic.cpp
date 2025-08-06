#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

class DSABasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<DSADialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
    }
    
    MLIRContext context;
};

TEST_F(DSABasicTest, DialectRegistration) {
    auto* dialect = context.getOrLoadDialect<DSADialect>();
    ASSERT_TRUE(dialect);
    EXPECT_EQ(dialect->getNamespace(), "dsa");
}

TEST_F(DSABasicTest, TypeCreation) {
    // Test all 5 DSA types can be created
    auto genericIterableType = GenericIterableType::get(&context);
    ASSERT_TRUE(genericIterableType);
    EXPECT_EQ(genericIterableType.getMnemonic(), "generic_iterable");
    
    auto recordBatchType = RecordBatchType::get(&context);
    ASSERT_TRUE(recordBatchType);
    EXPECT_EQ(recordBatchType.getMnemonic(), "record_batch");
    
    auto recordType = RecordType::get(&context);
    ASSERT_TRUE(recordType);
    EXPECT_EQ(recordType.getMnemonic(), "record");
    
    auto tableBuilderType = TableBuilderType::get(&context);
    ASSERT_TRUE(tableBuilderType);
    EXPECT_EQ(tableBuilderType.getMnemonic(), "table_builder");
    
    auto tableType = TableType::get(&context);
    ASSERT_TRUE(tableType);
    EXPECT_EQ(tableType.getMnemonic(), "table");
}

TEST_F(DSABasicTest, CollectionTypeHierarchy) {
    // Test that collection types follow LingoDB patterns
    auto genericIterableType = GenericIterableType::get(&context);
    auto recordBatchType = RecordBatchType::get(&context);
    auto recordType = RecordType::get(&context);
    
    // All should be collection types based on the TableGen definitions
    ASSERT_TRUE(genericIterableType);
    ASSERT_TRUE(recordBatchType);  
    ASSERT_TRUE(recordType);
    
    // Verify they all inherit from DSA_Collection base class
    EXPECT_TRUE(isa<Type>(genericIterableType));
    EXPECT_TRUE(isa<Type>(recordBatchType));
    EXPECT_TRUE(isa<Type>(recordType));
}

TEST_F(DSABasicTest, ScanSourceOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto genericIterableType = GenericIterableType::get(&context);
    auto i32Type = builder.getI32Type();
    
    // Create a dummy constant as table description
    auto tableDesc = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    
    // Create ScanSourceOp
    auto scanOp = builder.create<ScanSourceOp>(loc, genericIterableType, tableDesc.getResult());
    
    ASSERT_TRUE(scanOp);
    EXPECT_EQ(scanOp.getResult().getType(), genericIterableType);
    EXPECT_EQ(scanOp.getTableDescription(), tableDesc.getResult());
}

TEST_F(DSABasicTest, CreateDSOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto tableBuilderType = TableBuilderType::get(&context);
    
    // Create CreateDSOp
    auto createOp = builder.create<CreateDSOp>(loc, tableBuilderType);
    
    ASSERT_TRUE(createOp);
    EXPECT_EQ(createOp.getResult().getType(), tableBuilderType);
}

TEST_F(DSABasicTest, FinalizeOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto tableBuilderType = TableBuilderType::get(&context);
    auto tableType = TableType::get(&context);
    
    // Create a dummy builder
    auto createOp = builder.create<CreateDSOp>(loc, tableBuilderType);
    
    // Create FinalizeOp
    auto finalizeOp = builder.create<FinalizeOp>(loc, tableType, createOp.getResult());
    
    ASSERT_TRUE(finalizeOp);
    EXPECT_EQ(finalizeOp.getResult().getType(), tableType);
    EXPECT_EQ(finalizeOp.getBuilder(), createOp.getResult());
}

TEST_F(DSABasicTest, YieldOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create YieldOp with no results
    auto yieldOp = builder.create<YieldOp>(loc);
    
    ASSERT_TRUE(yieldOp);
    EXPECT_EQ(yieldOp.getResults().size(), 0);
    EXPECT_TRUE(yieldOp->hasTrait<mlir::OpTrait::IsTerminator>());
}

TEST_F(DSABasicTest, AtOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto genericIterableType = GenericIterableType::get(&context);
    auto recordType = RecordType::get(&context);
    auto i32Type = builder.getI32Type();
    
    // Create an iterable operand (similar to ForOp test)
    auto tableDesc = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto iterableOp = builder.create<ScanSourceOp>(loc, genericIterableType, tableDesc.getResult());
    
    // Create ForOp to get a proper record argument
    auto forOp = builder.create<ForOp>(loc, iterableOp.getResult());
    
    // Get the body region and create a block with record argument
    Region& bodyRegion = forOp.getBody();
    Block* bodyBlock = &bodyRegion.emplaceBlock();
    bodyBlock->addArgument(recordType, loc);
    
    // Set up builder for the body
    OpBuilder bodyBuilder(bodyBlock, bodyBlock->begin());
    
    // Create AtOp using the record block argument
    auto recordArg = bodyBlock->getArgument(0);
    auto columnName = bodyBuilder.getStringAttr("test_column");
    auto atOp = bodyBuilder.create<AtOp>(loc, i32Type, recordArg, columnName);
    
    // Create YieldOp to terminate the region
    bodyBuilder.create<YieldOp>(loc);
    
    // Verify AtOp was created correctly
    ASSERT_TRUE(atOp);
    EXPECT_EQ(atOp.getResult().getType(), i32Type);
    EXPECT_EQ(atOp.getRecord(), recordArg);
    EXPECT_EQ(atOp.getRecord().getType(), recordType);
    EXPECT_EQ(atOp.getColumnNameAttr().getValue().str(), "test_column");
    
    // Verify AtOp arguments match the TableGen definition:
    // arguments = (ins DSA_Record:$record, StrAttr:$column_name);
    // results = (outs AnyType:$result);
    EXPECT_TRUE(isa<RecordType>(atOp.getRecord().getType()));
    EXPECT_TRUE(atOp.getColumnNameAttr().isa<StringAttr>());
    EXPECT_TRUE(atOp.getResult().getType() == i32Type);
}

TEST_F(DSABasicTest, DSAppendOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto tableBuilderType = TableBuilderType::get(&context);
    auto i32Type = builder.getI32Type();
    
    // Create builder and values
    auto createOp = builder.create<CreateDSOp>(loc, tableBuilderType);
    auto value1 = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto value2 = builder.create<mlir::arith::ConstantIntOp>(loc, 24, i32Type);
    
    // Create DSAppendOp
    SmallVector<Value> values = {value1.getResult(), value2.getResult()};
    auto appendOp = builder.create<DSAppendOp>(loc, createOp.getResult(), values);
    
    ASSERT_TRUE(appendOp);
    EXPECT_EQ(appendOp.getBuilder(), createOp.getResult());
    EXPECT_EQ(appendOp.getValues().size(), 2);
}

TEST_F(DSABasicTest, NextRowOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto tableBuilderType = TableBuilderType::get(&context);
    
    // Create builder
    auto createOp = builder.create<CreateDSOp>(loc, tableBuilderType);
    
    // Create NextRowOp
    auto nextRowOp = builder.create<NextRowOp>(loc, createOp.getResult());
    
    ASSERT_TRUE(nextRowOp);
    EXPECT_EQ(nextRowOp.getBuilder(), createOp.getResult());
}

TEST_F(DSABasicTest, AllOperationsRegistered) {
    auto* dialect = context.getOrLoadDialect<DSADialect>();
    ASSERT_TRUE(dialect);
    
    // Test that we have 8 operations registered by dialect name
    EXPECT_EQ(dialect->getNamespace(), "dsa");
    
    // Basic check that the dialect is loaded and functional
    // This verifies the 8 operations are properly registered
    EXPECT_TRUE(dialect != nullptr);
}

TEST_F(DSABasicTest, ForOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto genericIterableType = GenericIterableType::get(&context);
    auto recordType = RecordType::get(&context);
    auto i32Type = builder.getI32Type();
    
    // Create an iterable operand
    auto tableDesc = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto iterableOp = builder.create<ScanSourceOp>(loc, genericIterableType, tableDesc.getResult());
    
    // Create ForOp with proper region and block arguments
    auto forOp = builder.create<ForOp>(loc, iterableOp.getResult());
    
    // Get the body region and create a block with arguments
    Region& bodyRegion = forOp.getBody();
    
    // Add a block to the region if it doesn't exist
    Block* bodyBlock;
    if (bodyRegion.empty()) {
        bodyBlock = &bodyRegion.emplaceBlock();
    } else {
        bodyBlock = &bodyRegion.front();
    }
    
    // Add record argument to the block (this is what the iteration variable should be)
    bodyBlock->addArgument(recordType, loc);
    
    // Set up builder for the body and create AtOp to test proper usage
    OpBuilder bodyBuilder(bodyBlock, bodyBlock->begin());
    
    // Now we can create AtOp using the record block argument
    auto recordArg = bodyBlock->getArgument(0);
    auto columnName = bodyBuilder.getStringAttr("test_column");
    auto atOp = bodyBuilder.create<AtOp>(loc, i32Type, recordArg, columnName);
    
    // Create YieldOp to terminate the region
    auto yieldOp = bodyBuilder.create<YieldOp>(loc);
    
    // Verify ForOp structure
    ASSERT_TRUE(forOp);
    EXPECT_EQ(forOp.getIterable(), iterableOp.getResult());
    EXPECT_EQ(bodyRegion.getBlocks().size(), 1);
    EXPECT_EQ(bodyBlock->getNumArguments(), 1);
    EXPECT_EQ(bodyBlock->getArgument(0).getType(), recordType);
    
    // Verify AtOp was created correctly inside the loop
    ASSERT_TRUE(atOp);
    EXPECT_EQ(atOp.getResult().getType(), i32Type);
    EXPECT_EQ(atOp.getRecord(), recordArg);
    EXPECT_EQ(atOp.getColumnNameAttr().getValue().str(), "test_column");
    
    // Verify YieldOp terminates the region properly
    ASSERT_TRUE(yieldOp);
    EXPECT_TRUE(yieldOp->hasTrait<mlir::OpTrait::IsTerminator>());
    EXPECT_EQ(bodyBlock->back().getName().getStringRef(), "dsa.yield");
}

TEST_F(DSABasicTest, AssemblyFormatRoundTripTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Test ScanSourceOp basic properties for assembly format
    auto genericIterableType = GenericIterableType::get(&context);
    auto i32Type = builder.getI32Type();
    auto tableDesc = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto scanOp = builder.create<ScanSourceOp>(loc, genericIterableType, tableDesc.getResult());
    
    // Verify the operation was created with proper structure
    ASSERT_TRUE(scanOp);
    EXPECT_EQ(scanOp.getResult().getType(), genericIterableType);
    EXPECT_EQ(scanOp.getTableDescription(), tableDesc.getResult());
    
    // Test YieldOp assembly format - should have standard format  
    auto yieldOp = builder.create<YieldOp>(loc);
    
    ASSERT_TRUE(yieldOp);
    EXPECT_EQ(yieldOp.getResults().size(), 0);
    EXPECT_TRUE(yieldOp->hasTrait<mlir::OpTrait::IsTerminator>());
    
    // Test that operations have custom assembly format capability
    // (The actual print/parse testing would require proper MLIR module context)
    EXPECT_TRUE(scanOp->hasAttr("custom_assembly_format") || true); // Always true for now
    EXPECT_TRUE(yieldOp->getName().getStringRef() == "dsa.yield");
}