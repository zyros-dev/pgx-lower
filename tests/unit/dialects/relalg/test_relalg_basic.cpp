#include <gtest/gtest.h>
#include "execution/logging.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

TEST(RelAlgBasicTest, DialectRegistration) {
    MLIRContext context;
    auto* dialect = context.getOrLoadDialect<RelAlgDialect>();
    ASSERT_TRUE(dialect);
    EXPECT_EQ(dialect->getNamespace(), "relalg");
}

TEST(RelAlgBasicTest, BasicBuildingBlocks) {
    // Just test that we can create a context without crashes
    MLIRContext context;
    context.getOrLoadDialect<RelAlgDialect>();
    
    // This mainly tests that TableGen generated code compiles
    EXPECT_TRUE(true);
}

TEST(RelAlgBasicTest, BaseTableOpWithTableOid) {
    PGX_DEBUG("Testing BaseTableOp with table_oid attribute");
    
    MLIRContext context;
    context.getOrLoadDialect<RelAlgDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    
    // Create a module and function
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_base_table", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Test BaseTableOp with table_oid attribute
    auto tableType = TupleStreamType::get(&context);
    auto baseTableOp = builder.create<BaseTableOp>(
        builder.getUnknownLoc(),
        tableType,
        builder.getStringAttr("customers"),
        builder.getI64IntegerAttr(12345));
    
    // BaseTableOp no longer has a body - it's a simple operation
    builder.setInsertionPointAfter(baseTableOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify the operation was created correctly
    ASSERT_TRUE(baseTableOp);
    EXPECT_EQ(baseTableOp.getTableName(), "customers");
    EXPECT_EQ(baseTableOp.getTableOid(), 12345);
    PGX_DEBUG("BaseTableOp created successfully with table_oid");
}

TEST(RelAlgBasicTest, MaterializeOpWithColumns) {
    PGX_DEBUG("Testing MaterializeOp with columns attribute");
    
    MLIRContext context;
    context.getOrLoadDialect<RelAlgDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    
    OpBuilder builder(&context);
    
    // Create a module and function
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_materialize", funcType);
    auto *funcBody = func.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);
    
    // Create a BaseTableOp as source for MaterializeOp
    auto tupleStreamType = TupleStreamType::get(&context);
    auto baseTableOp = builder.create<BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("orders"),
        builder.getI64IntegerAttr(67890));
    
    // BaseTableOp no longer has a body - it's a simple operation
    // Test MaterializeOp with columns attribute
    builder.setInsertionPointAfter(baseTableOp);
    SmallVector<Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("order_id"));
    columnAttrs.push_back(builder.getStringAttr("customer_id"));
    columnAttrs.push_back(builder.getStringAttr("order_date"));
    columnAttrs.push_back(builder.getStringAttr("total_amount"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<MaterializeOp>(
        builder.getUnknownLoc(),
        TableType::get(&context),
        baseTableOp.getResult(),
        columnsArrayAttr);
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify the operation was created correctly
    ASSERT_TRUE(materializeOp);
    
    // Verify columns attribute
    auto columns = materializeOp.getColumns();
    ASSERT_EQ(columns.size(), 4);
    EXPECT_EQ(cast<StringAttr>(columns[0]).getValue(), "order_id");
    EXPECT_EQ(cast<StringAttr>(columns[1]).getValue(), "customer_id");
    EXPECT_EQ(cast<StringAttr>(columns[2]).getValue(), "order_date");
    EXPECT_EQ(cast<StringAttr>(columns[3]).getValue(), "total_amount");
    
    PGX_DEBUG("MaterializeOp created successfully with columns attribute");
}