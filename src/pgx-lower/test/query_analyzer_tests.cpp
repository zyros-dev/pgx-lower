extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "utils/fmgroids.h"
}

#include "pgx-lower/frontend/SQL/query_analyzer.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

auto makeIntConst() -> Const {
    auto value = Const{};
    value.xpr.type = T_Const;
    value.consttype = INT4OID;
    value.consttypmod = -1;
    value.constisnull = false;
    value.constbyval = true;
    value.constlen = sizeof(int32);
    value.constvalue = Datum{1};
    return value;
}

} // namespace

PGX_TEST_FN(query_analyzer_default_result_is_invalid) {
    const auto result = pgx_lower::AnalyzerResult{};
    REQUIRE(!result.isSupported());
    REQUIRE(result.reasons().size() == 1);
    REQUIRE(result.reasons().front().kind == pgx_lower::UnsupportedReasonKind::invalid);
    REQUIRE(result.primaryReasonKindName() == std::string("invalid"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_supported_result_has_no_reasons) {
    const auto result = pgx_lower::AnalyzerResult::supported();
    REQUIRE(result.isSupported());
    REQUIRE(result.reasons().empty());
    REQUIRE(result.humanSummary() == std::string("supported"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_unsupported_result_reports_first_reason) {
    auto result = pgx_lower::AnalyzerResult::unsupported(pgx_lower::UnsupportedReasonKind::unsupported_function,
                                                         "unsupported function generate_series()",
                                                         "Plan.Result.targetlist[0]");
    result.addUnsupportedReason(pgx_lower::UnsupportedReasonKind::unsupported_type, "unsupported type jsonb",
                                "Plan.Result.targetlist[1]");

    REQUIRE(!result.isSupported());
    REQUIRE(result.reasons().size() == 2);
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    REQUIRE(result.primaryReasonKindName() == std::string("unsupported_function"));
    REQUIRE(result.humanSummary()
            == std::string("unsupported_function: unsupported function generate_series() at "
                           "Plan.Result.targetlist[0]"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unknown_plan_node) {
    auto plan = Plan{};
    plan.type = T_Invalid;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(&plan);
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unknown_expr_node) {
    auto expr = Node{};
    expr.type = T_Invalid;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(&expr);
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_translator_unsupported_result_plan) {
    auto plan = Plan{};
    plan.type = T_Result;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(&plan);
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_translator_unsupported_set_operation_plan) {
    auto plan = Plan{};
    plan.type = T_SetOp;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(&plan);
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_function_signature_mismatch) {
    auto arg = makeIntConst();
    auto func = FuncExpr{};
    func.xpr.type = T_FuncExpr;
    func.funcid = F_UPPER_TEXT;
    func.funcresulttype = TEXTOID;
    func.inputcollid = InvalidOid;
    func.funccollid = InvalidOid;
    func.args = list_make1(&arg);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&func));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_operator_signature_mismatch) {
    auto lhs = makeIntConst();
    auto rhs = makeIntConst();
    auto op = OpExpr{};
    op.xpr.type = T_OpExpr;
    op.opno = TextEqualOperator;
    op.opfuncid = F_TEXTEQ;
    op.opresulttype = BOOLOID;
    op.inputcollid = InvalidOid;
    op.opcollid = InvalidOid;
    op.args = list_make2(&lhs, &rhs);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_target_expr_metadata) {
    auto target = TargetEntry{};
    target.xpr.type = T_TargetEntry;
    target.expr = nullptr;
    target.resjunk = false;

    auto plan = Plan{};
    plan.type = T_SeqScan;
    plan.targetlist = list_make1(&target);

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(&plan);
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_root_targetlist_metadata) {
    auto plan = Plan{};
    plan.type = T_SeqScan;
    plan.targetlist = nullptr;

    auto stmt = PlannedStmt{};
    stmt.commandType = CMD_SELECT;
    stmt.planTree = &plan;

    const auto result = pgx_lower::QueryAnalyzer::analyzePlan(&stmt);
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}
