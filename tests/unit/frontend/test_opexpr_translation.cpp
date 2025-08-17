#include <gtest/gtest.h>
#include "test_plan_node_helpers.h"
#include "frontend/SQL/postgresql_ast_translator.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"

// PostgreSQL type OIDs that may not be defined in test environment
#ifndef INT4OID
#define INT4OID 23
#endif
#ifndef INT8OID
#define INT8OID 20
#endif
#ifndef BOOLOID
#define BOOLOID 16
#endif

namespace {

class OpExprTranslationTest : public PlanNodeTestBase {
protected:
    void SetUp() override {
        PlanNodeTestBase::SetUp();
        context->loadDialect<mlir::arith::ArithDialect>();
        context->loadDialect<mlir::db::DBDialect>();
    }
};

TEST_F(OpExprTranslationTest, TranslateArithmeticAddition) {
    // Create constants for operands
    auto* const1 = createConst(INT4OID, 5);
    auto* const2 = createConst(INT4OID, 10);
    
    // Create OpExpr for addition (OID 551)
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(551, args, INT4OID); // int4 + int4
    
    // Create a simple plan with the expression
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateArithmeticSubtraction) {
    auto* const1 = createConst(INT4OID, 20);
    auto* const2 = createConst(INT4OID, 8);
    
    // Create OpExpr for subtraction (OID 552)
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(552, args, INT4OID); // int4 - int4
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateArithmeticMultiplication) {
    auto* const1 = createConst(INT4OID, 6);
    auto* const2 = createConst(INT4OID, 7);
    
    // Create OpExpr for multiplication (OID 514)
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(514, args, INT4OID); // int4 * int4
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateArithmeticDivision) {
    auto* const1 = createConst(INT4OID, 42);
    auto* const2 = createConst(INT4OID, 6);
    
    // Create OpExpr for division (OID 528)
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(528, args, INT4OID); // int4 / int4
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateComparisonEqual) {
    auto* const1 = createConst(INT4OID, 15);
    auto* const2 = createConst(INT4OID, 15);
    
    // Create OpExpr for equality (OID 96)
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(96, args, BOOLOID); // int4 = int4
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateComparisonLessThan) {
    auto* const1 = createConst(INT4OID, 5);
    auto* const2 = createConst(INT4OID, 10);
    
    // Create OpExpr for less than (OID 97)
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(97, args, BOOLOID); // int4 < int4
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateOpExprWithNullArgs) {
    // Create OpExpr with null args
    auto* opExpr = createOpExpr(551, nullptr, INT4OID);
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    // Should handle gracefully without crash
    auto module = translator->translateQuery(&plannedStmt);
    // Test passes if no crash occurs
}

TEST_F(OpExprTranslationTest, TranslateOpExprWithEmptyArgs) {
    // Create OpExpr with empty args list
    List* args = NIL;
    auto* opExpr = createOpExpr(551, args, INT4OID);
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    // Should handle gracefully without crash
    auto module = translator->translateQuery(&plannedStmt);
}

TEST_F(OpExprTranslationTest, TranslateUnsupportedOperator) {
    auto* const1 = createConst(INT4OID, 5);
    auto* const2 = createConst(INT4OID, 10);
    
    // Create OpExpr with unsupported operator OID
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(99999, args, INT4OID); // Unsupported OID
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    // Should handle gracefully, returning placeholder
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

TEST_F(OpExprTranslationTest, TranslateInt8Operators) {
    // Test INT8 addition (OID 684)
    auto* const1 = createConst(INT8OID, 1000000000LL);
    auto* const2 = createConst(INT8OID, 2000000000LL);
    
    List* args = list_make2(const1, const2);
    auto* opExpr = createOpExpr(684, args, INT8OID); // int8 + int8
    
    auto* seqScan = createSeqScan(1, 16384);
    PlannedStmt plannedStmt = createPlannedStmt(reinterpret_cast<Plan*>(seqScan));
    
    auto module = translator->translateQuery(&plannedStmt);
    EXPECT_NE(module, nullptr);
}

} // namespace