#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <dialects/pg/PgDialect.h>

class PgDialectBasicTest : public ::testing::Test {
   protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        dialect = context->getOrLoadDialect<pgx_lower::compiler::dialect::pg::PgDialect>();
    }

    std::unique_ptr<mlir::MLIRContext> context;
    pgx_lower::compiler::dialect::pg::PgDialect* dialect = nullptr;
};

TEST_F(PgDialectBasicTest, DialectRegistration) {
    // Test basic pg dialect registration
    EXPECT_NE(dialect, nullptr);
    EXPECT_EQ(dialect->getNamespace(), "pg");
}

TEST_F(PgDialectBasicTest, BasicTypeCreation) {
    // Test basic type creation
    const auto textType = pgx_lower::compiler::dialect::pg::TextType::get(context.get());
    const auto tableHandleType = pgx_lower::compiler::dialect::pg::TableHandleType::get(context.get());
    const auto tupleHandleType = pgx_lower::compiler::dialect::pg::TupleHandleType::get(context.get());

    EXPECT_TRUE(textType);
    EXPECT_TRUE(tableHandleType);
    EXPECT_TRUE(tupleHandleType);

    // Test type properties
    EXPECT_TRUE(mlir::isa<pgx_lower::compiler::dialect::pg::TextType>(textType));
    EXPECT_TRUE(mlir::isa<pgx_lower::compiler::dialect::pg::TableHandleType>(tableHandleType));
    EXPECT_TRUE(mlir::isa<pgx_lower::compiler::dialect::pg::TupleHandleType>(tupleHandleType));
}

TEST_F(PgDialectBasicTest, ParametricTypeCreation) {
    // Test parametric types
    const auto numericType = pgx_lower::compiler::dialect::pg::NumericType::get(context.get(), 10, 2);
    EXPECT_TRUE(numericType);
    EXPECT_EQ(numericType.getPrecision(), 10);
    EXPECT_EQ(numericType.getScale(), 2);

    const auto charType = pgx_lower::compiler::dialect::pg::CharType::get(context.get(), 50);
    EXPECT_TRUE(charType);
    EXPECT_EQ(charType.getLength(), 50);

    // Test different numeric precisions
    const auto numericType2 = pgx_lower::compiler::dialect::pg::NumericType::get(context.get(), 5, 1);
    EXPECT_EQ(numericType2.getPrecision(), 5);
    EXPECT_EQ(numericType2.getScale(), 1);

    // Test different char lengths
    auto charType2 = pgx_lower::compiler::dialect::pg::CharType::get(context.get(), 100);
    EXPECT_EQ(charType2.getLength(), 100);
}