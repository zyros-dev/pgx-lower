#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"

class DBConstantCharTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    std::unique_ptr<mlir::ModuleOp> module;
    mlir::OpBuilder builder;

    DBConstantCharTest() : builder(&context) {
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(
            mlir::UnknownLoc::get(&context)));
    }
};

TEST_F(DBConstantCharTest, CharConstantCreation) {
    // Test that we can create a db.constant with char type
    auto charType = mlir::db::CharType::get(&context, 5);
    auto constantOp = builder.create<mlir::db::ConstantOp>(
        builder.getUnknownLoc(), 
        charType,
        builder.getStringAttr("hello"));
    
    ASSERT_TRUE(constantOp) << "Should be able to create db.constant with char type";
    EXPECT_EQ(constantOp.getType(), charType) << "Constant should have char type";
    
    auto stringAttr = constantOp.getConstantValue().dyn_cast<mlir::StringAttr>();
    ASSERT_TRUE(stringAttr) << "Constant value should be a string attribute";
    EXPECT_EQ(stringAttr.getValue(), "hello") << "Constant value should be 'hello'";
}

TEST_F(DBConstantCharTest, CharConstantWithEmptyString) {
    // Test creating a char constant with an empty string
    auto charType = mlir::db::CharType::get(&context, 3);
    auto constantOp = builder.create<mlir::db::ConstantOp>(
        builder.getUnknownLoc(), 
        charType,
        builder.getStringAttr(""));
    
    ASSERT_TRUE(constantOp) << "Should be able to create db.constant with empty string";
    EXPECT_EQ(constantOp.getType(), charType) << "Constant should have char type";
    
    auto stringAttr = constantOp.getConstantValue().dyn_cast<mlir::StringAttr>();
    ASSERT_TRUE(stringAttr) << "Constant value should be a string attribute";
    EXPECT_EQ(stringAttr.getValue(), "") << "Constant value should be empty";
}

TEST_F(DBConstantCharTest, CharConstantWith8Bytes) {
    // Test creating a char constant with 8-byte string (max supported)
    auto charType = mlir::db::CharType::get(&context, 8);
    auto constantOp = builder.create<mlir::db::ConstantOp>(
        builder.getUnknownLoc(), 
        charType,
        builder.getStringAttr("12345678"));
    
    ASSERT_TRUE(constantOp) << "Should be able to create 8-byte char constant";
    EXPECT_EQ(constantOp.getType(), charType) << "Constant should have char<8> type";
    
    auto stringAttr = constantOp.getConstantValue().dyn_cast<mlir::StringAttr>();
    ASSERT_TRUE(stringAttr) << "Constant value should be a string attribute";
    EXPECT_EQ(stringAttr.getValue(), "12345678") << "Constant value should be '12345678'";
}