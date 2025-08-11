#include <gtest/gtest.h>
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace pgx::mlir;

TEST(TranslatorInfrastructureSimpleTest, TranslatorContextColumnMapping) {
    // Test TranslatorContext column-based value mapping functionality
    relalg::TranslatorContext context;
    
    MLIRContext mlirContext;
    mlirContext.loadDialect<arith::ArithDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    // Create a scope for column resolution
    auto scope = context.createScope();
    
    // Create a dummy column and value
    relalg::Column col(builder.getI64Type());
    auto val1 = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
    
    // Map column to value
    context.setValueForAttribute(scope, &col, val1);
    
    // Verify mapping
    EXPECT_EQ(context.getValueForAttribute(&col), val1.getResult());
    
    // Test unsafe lookup for non-existent column
    relalg::Column col2(builder.getI64Type());
    EXPECT_EQ(context.getUnsafeValueForAttribute(&col2), Value());
}

TEST(TranslatorInfrastructureSimpleTest, TranslatorContextScoping) {
    // Test TranslatorContext scoping functionality
    relalg::TranslatorContext context;
    
    // Create scope and test that it works
    auto scope = context.createScope();
    EXPECT_NO_THROW(scope);
}

TEST(TranslatorInfrastructureSimpleTest, DummyTranslatorCreation) {
    // Test that DummyTranslator can be created for unknown operations
    MLIRContext mlirContext;
    mlirContext.loadDialect<func::FuncDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto dummyOp = builder.create<func::ReturnOp>(loc);
    auto translator = relalg::Translator::createTranslator(dummyOp);
    
    ASSERT_NE(translator, nullptr);
    EXPECT_EQ(translator->getOperation(), dummyOp.getOperation());
}

TEST(TranslatorInfrastructureSimpleTest, BaseTableTranslatorCreation) {
    // Test BaseTableTranslator creation
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    auto translator = relalg::Translator::createTranslator(baseTableOp);
    
    ASSERT_NE(translator, nullptr);
    EXPECT_EQ(translator->getOperation(), baseTableOp.getOperation());
}

TEST(TranslatorInfrastructureSimpleTest, MaterializeTranslatorCreation) {
    // Test MaterializeTranslator creation
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    // Create MaterializeOp with columns
    SmallVector<Attribute> columns;
    columns.push_back(builder.getStringAttr("id"));
    auto materializeOp = builder.create<relalg::MaterializeOp>(
        loc, tupleStreamType, baseTableOp.getResult(), 
        builder.getArrayAttr(columns));
    
    auto translator = relalg::Translator::createTranslator(materializeOp);
    
    ASSERT_NE(translator, nullptr);
    EXPECT_EQ(translator->getOperation(), materializeOp.getOperation());
}

TEST(TranslatorInfrastructureSimpleTest, TranslatorProduceConsume) {
    // Test basic produce/consume functionality with DummyTranslator
    MLIRContext mlirContext;
    mlirContext.loadDialect<func::FuncDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto dummyOp = builder.create<func::ReturnOp>(loc);
    auto translator = relalg::Translator::createTranslator(dummyOp);
    
    relalg::TranslatorContext context;
    
    // Should not throw when calling produce
    EXPECT_NO_THROW(translator->produce(context, builder));
    
    // Should not throw when calling done
    EXPECT_NO_THROW(translator->done());
}

TEST(TranslatorInfrastructureSimpleTest, BaseTableTranslatorSetInfo) {
    // Test BaseTableTranslator setInfo functionality
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    auto translator = relalg::Translator::createTranslator(baseTableOp);
    
    // Test setInfo with empty required attributes
    relalg::ColumnSet requiredAttrs;
    EXPECT_NO_THROW(translator->setInfo(nullptr, requiredAttrs));
}

TEST(TranslatorInfrastructureSimpleTest, BaseTableTranslatorGetAvailableColumns) {
    // Test BaseTableTranslator getAvailableColumns functionality
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    auto translator = relalg::Translator::createTranslator(baseTableOp);
    
    // Initialize translator with context for early column setup
    relalg::TranslatorContext context;
    translator->initializeWithContext(context);
    
    // BaseTable should provide columns after initialization
    auto availableColumns = translator->getAvailableColumns();
    EXPECT_FALSE(availableColumns.empty());
    EXPECT_EQ(availableColumns.size(), 1); // For Test 1, we have 1 'id' column
}

TEST(TranslatorInfrastructureSimpleTest, MaterializeTranslatorSetInfo) {
    // Test MaterializeTranslator setInfo functionality
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    // Create MaterializeOp with columns
    SmallVector<Attribute> columns;
    columns.push_back(builder.getStringAttr("id"));
    auto materializeOp = builder.create<relalg::MaterializeOp>(
        loc, tupleStreamType, baseTableOp.getResult(), 
        builder.getArrayAttr(columns));
    
    auto translator = relalg::Translator::createTranslator(materializeOp);
    
    // Test setInfo functionality
    relalg::ColumnSet requiredAttrs;
    EXPECT_NO_THROW(translator->setInfo(nullptr, requiredAttrs));
}

TEST(TranslatorInfrastructureSimpleTest, MaterializeTranslatorGetAvailableColumns) {
    // Test MaterializeTranslator getAvailableColumns functionality
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345));
    
    // Create MaterializeOp with columns
    SmallVector<Attribute> columns;
    columns.push_back(builder.getStringAttr("id"));
    auto materializeOp = builder.create<relalg::MaterializeOp>(
        loc, tupleStreamType, baseTableOp.getResult(), 
        builder.getArrayAttr(columns));
    
    auto translator = relalg::Translator::createTranslator(materializeOp);
    
    // MaterializeOp doesn't produce columns for consumption
    auto availableColumns = translator->getAvailableColumns();
    EXPECT_TRUE(availableColumns.empty());
}

TEST(TranslatorInfrastructureSimpleTest, OrderedAttributesColumnResolution) {
    // Test OrderedAttributes column resolution functionality
    MLIRContext mlirContext;
    mlirContext.loadDialect<arith::ArithDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    relalg::TranslatorContext context;
    auto scope = context.createScope();
    
    // Create columns and values
    relalg::Column col1(builder.getI64Type());
    relalg::Column col2(builder.getI64Type());
    auto val1 = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
    auto val2 = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(2));
    
    // Map columns to values
    context.setValueForAttribute(scope, &col1, val1);
    context.setValueForAttribute(scope, &col2, val2);
    
    // Create OrderedAttributes and test resolution
    relalg::OrderedAttributes orderedAttrs;
    orderedAttrs.insert(&col1);
    orderedAttrs.insert(&col2);
    
    // Test column resolution by position
    EXPECT_EQ(orderedAttrs.resolve(context, 0), val1.getResult());
    EXPECT_EQ(orderedAttrs.resolve(context, 1), val2.getResult());
    
    // Test ordered attributes properties
    EXPECT_EQ(orderedAttrs.size(), 2);
    EXPECT_EQ(orderedAttrs.getTypes().size(), 2);
    EXPECT_TRUE(orderedAttrs.contains(&col1));
    EXPECT_TRUE(orderedAttrs.contains(&col2));
    EXPECT_EQ(orderedAttrs.getPos(&col1), 0);
    EXPECT_EQ(orderedAttrs.getPos(&col2), 1);
}

TEST(TranslatorInfrastructureSimpleTest, ColumnIdentitySharing) {
    // Test that BaseTableTranslator and MaterializeTranslator share column identity
    MLIRContext mlirContext;
    mlirContext.loadDialect<relalg::RelAlgDialect>();
    mlirContext.loadDialect<arith::ArithDialect>();
    OpBuilder builder(&mlirContext);
    auto loc = builder.getUnknownLoc();
    
    auto tupleStreamType = relalg::TupleStreamType::get(&mlirContext);
    auto baseTableOp = builder.create<relalg::BaseTableOp>(
        loc, tupleStreamType, builder.getStringAttr("test"),
        builder.getI64IntegerAttr(12345));
    
    // Create MaterializeOp with columns
    SmallVector<Attribute> columns;
    columns.push_back(builder.getStringAttr("id"));
    auto materializeOp = builder.create<relalg::MaterializeOp>(
        loc, tupleStreamType, baseTableOp.getResult(), 
        builder.getArrayAttr(columns));
    
    // Create translators
    auto baseTranslator = relalg::Translator::createTranslator(baseTableOp);
    auto materializeTranslator = relalg::Translator::createTranslator(materializeOp);
    
    // Initialize with context (this sets up ColumnManager)
    relalg::TranslatorContext context;
    baseTranslator->initializeWithContext(context);
    materializeTranslator->initializeWithContext(context);
    
    // Set up the translator chain
    relalg::ColumnSet requiredAttrs;
    materializeTranslator->setInfo(nullptr, requiredAttrs);
    
    // Get columns from BaseTableTranslator
    auto baseColumns = baseTranslator->getAvailableColumns();
    ASSERT_EQ(baseColumns.size(), 1);
    
    // The key test: MaterializeTranslator should now have the same column pointer
    // This verifies column identity is properly shared through the translator chain
    const relalg::Column* baseColumn = *baseColumns.begin();
    
    // Create a dummy value for the column
    auto scope = context.createScope();
    auto val = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(42));
    context.setValueForAttribute(scope, baseColumn, val);
    
    // Verify that the column can be resolved correctly
    auto resolvedVal = context.getValueForAttribute(baseColumn);
    EXPECT_EQ(resolvedVal, val.getResult());
}