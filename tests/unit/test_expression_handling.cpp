#include <gtest/gtest.h>
#include "core/mlir_builder.h"
#include "core/mlir_logger.h"
#include "mlir/IR/MLIRContext.h"
#include <vector>

using namespace mlir_builder;

class ExpressionHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        logger = std::make_unique<ConsoleLogger>();
        builder = createMLIRBuilder(*context);
    }

    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<MLIRLogger> logger;
    std::unique_ptr<MLIRBuilder> builder;
};

TEST_F(ExpressionHandlingTest, ColumnExpressionBasicConstructors) {
    // Test regular column constructor
    ColumnExpression col1(0);
    EXPECT_EQ(col1.columnIndex, 0);
    EXPECT_TRUE(col1.operatorName.empty());
    EXPECT_TRUE(col1.operandColumns.empty());

    ColumnExpression col2(5);
    EXPECT_EQ(col2.columnIndex, 5);
}

TEST_F(ExpressionHandlingTest, ColumnExpressionArithmeticConstructors) {
    // Test arithmetic expression constructor
    std::vector<int> operands = {0, 1};
    ColumnExpression addExpr("+", operands);
    
    EXPECT_EQ(addExpr.columnIndex, -1);  // Computed expression marker
    EXPECT_EQ(addExpr.operatorName, "+");
    EXPECT_EQ(addExpr.operandColumns.size(), 2);
    EXPECT_EQ(addExpr.operandColumns[0], 0);
    EXPECT_EQ(addExpr.operandColumns[1], 1);
}

TEST_F(ExpressionHandlingTest, ArithmeticOperatorTypes) {
    std::vector<int> operands = {0, 1};
    
    // Test all supported arithmetic operators
    ColumnExpression addExpr("+", operands);
    EXPECT_EQ(addExpr.operatorName, "+");
    
    ColumnExpression subExpr("-", operands);
    EXPECT_EQ(subExpr.operatorName, "-");
    
    ColumnExpression mulExpr("*", operands);
    EXPECT_EQ(mulExpr.operatorName, "*");
    
    ColumnExpression divExpr("/", operands);
    EXPECT_EQ(divExpr.operatorName, "/");
    
    ColumnExpression modExpr("%", operands);
    EXPECT_EQ(modExpr.operatorName, "%");
}

TEST_F(ExpressionHandlingTest, MixedColumnAndExpressionVector) {
    std::vector<ColumnExpression> expressions;
    
    // Add regular columns
    expressions.emplace_back(0);  // val1
    expressions.emplace_back(1);  // val2
    
    // Add arithmetic expression: val1 + val2
    std::vector<int> operands = {0, 1};
    expressions.emplace_back("+", operands);
    
    // Verify the vector contents
    EXPECT_EQ(expressions.size(), 3);
    EXPECT_EQ(expressions[0].columnIndex, 0);
    EXPECT_EQ(expressions[1].columnIndex, 1);
    EXPECT_EQ(expressions[2].columnIndex, -1);
    EXPECT_EQ(expressions[2].operatorName, "+");
}

TEST_F(ExpressionHandlingTest, MLIRModuleGeneration) {
    // Test that we can create an MLIR module with mixed expressions
    std::vector<ColumnExpression> expressions;
    expressions.emplace_back(0);  // Regular column
    
    std::vector<int> operands = {0, 1};
    expressions.emplace_back("+", operands);  // Arithmetic expression
    
    // This should not crash and should return a valid module
    auto module = builder->buildTableScanModule("test_table", expressions);
    EXPECT_NE(module, nullptr);
    
    // The module should be valid MLIR
    EXPECT_TRUE(module->verify().succeeded());
}

TEST_F(ExpressionHandlingTest, ComplexArithmeticExpression) {
    std::vector<ColumnExpression> expressions;
    
    // Simulate: SELECT val1, val2, val1 + val2, val1 * val2 FROM table
    expressions.emplace_back(0);  // val1
    expressions.emplace_back(1);  // val2
    
    std::vector<int> addOperands = {0, 1};
    expressions.emplace_back("+", addOperands);  // val1 + val2
    
    std::vector<int> mulOperands = {0, 1};
    expressions.emplace_back("*", mulOperands);  // val1 * val2
    
    EXPECT_EQ(expressions.size(), 4);
    EXPECT_EQ(expressions[2].operatorName, "+");
    EXPECT_EQ(expressions[3].operatorName, "*");
    
    // Verify MLIR generation works
    auto module = builder->buildTableScanModule("test_table", expressions);
    EXPECT_NE(module, nullptr);
    EXPECT_TRUE(module->verify().succeeded());
}

TEST_F(ExpressionHandlingTest, ConstantOperandSupport) {
    // Test constructor with constants
    std::vector<int> operands = {0};  // One column operand
    std::vector<int> constants = {42}; // One constant operand
    ColumnExpression expr("+", operands, constants);
    
    EXPECT_EQ(expr.operatorName, "+");
    EXPECT_EQ(expr.operandColumns.size(), 1);
    EXPECT_EQ(expr.operandConstants.size(), 1);
    EXPECT_EQ(expr.operandConstants[0], 42);
}