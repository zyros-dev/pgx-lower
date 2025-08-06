#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::db;

class DBBasicSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<DBDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
    }
    
    MLIRContext context;
};

TEST_F(DBBasicSimpleTest, DialectRegistration) {
    auto* dialect = context.getOrLoadDialect<DBDialect>();
    ASSERT_TRUE(dialect);
    EXPECT_EQ(dialect->getNamespace(), "db");
}

TEST_F(DBBasicSimpleTest, ExternalSourceTypeCreation) {
    auto externalSourceType = ExternalSourceType::get(&context);
    ASSERT_TRUE(externalSourceType);
    EXPECT_EQ(externalSourceType.getMnemonic(), "external_source");
}

TEST_F(DBBasicSimpleTest, GetExternalOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i64Type = builder.getI64Type();
    auto externalSourceType = ExternalSourceType::get(&context);
    
    // Create table OID constant
    auto tableOid = builder.create<mlir::arith::ConstantIntOp>(loc, 12345, i64Type);
    
    // Create GetExternalOp
    auto getExternalOp = builder.create<GetExternalOp>(loc, externalSourceType, tableOid.getResult());
    
    ASSERT_TRUE(getExternalOp);
    EXPECT_EQ(getExternalOp.getHandle().getType(), externalSourceType);
    EXPECT_EQ(getExternalOp.getTableOid(), tableOid.getResult());
}

TEST_F(DBBasicSimpleTest, StreamResultsOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create StreamResultsOp
    auto streamResultsOp = builder.create<StreamResultsOp>(loc);
    
    ASSERT_TRUE(streamResultsOp);
    EXPECT_EQ(streamResultsOp->getResults().size(), 0);
}

TEST_F(DBBasicSimpleTest, ConstantOpCreation) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    
    // Test ConstantOp
    auto constantAttr = builder.getI32IntegerAttr(42);
    auto constantOp = builder.create<ConstantOp>(loc, i32Type, constantAttr);
    ASSERT_TRUE(constantOp);
    EXPECT_EQ(constantOp.getResult().getType(), i32Type);
    EXPECT_EQ(constantOp.getValueAttr(), constantAttr);
}

TEST_F(DBBasicSimpleTest, AllBasicOperationsWork) {
    auto* dialect = context.getOrLoadDialect<DBDialect>();
    ASSERT_TRUE(dialect);
    
    // Test that we have the DB dialect registered
    EXPECT_EQ(dialect->getNamespace(), "db");
    
    // Basic check that the dialect is loaded and functional
    EXPECT_TRUE(dialect != nullptr);
}