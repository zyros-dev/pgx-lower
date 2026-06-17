extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
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

constexpr Oid Int4LessEqualOperator = 523;
constexpr Oid Int4GreaterOperator = 521;

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

auto makeBoolConst() -> Const {
    auto value = Const{};
    value.xpr.type = T_Const;
    value.consttype = BOOLOID;
    value.consttypmod = -1;
    value.constisnull = false;
    value.constbyval = true;
    value.constlen = sizeof(bool);
    value.constvalue = BoolGetDatum(true);
    return value;
}

auto makeTypedConst(Oid typeOid) -> Const {
    auto value = Const{};
    value.xpr.type = T_Const;
    value.consttype = typeOid;
    value.consttypmod = -1;
    value.constcollid = OidIsValid(typeOid) && (typeOid == TEXTOID || typeOid == VARCHAROID || typeOid == BPCHAROID)
                            ? DEFAULT_COLLATION_OID
                            : InvalidOid;
    value.constisnull = false;
    value.constbyval = false;
    value.constlen = -1;
    value.constvalue = Datum{0};
    return value;
}

struct SortPlanFixture {
    Const value{};
    TargetEntry target{};
    Sort sort{};
    AttrNumber sortColIdx[1]{1};
    Oid sortOperators[1]{Int4LessOperator};
    Oid collations[1]{InvalidOid};
    bool nullsFirst[1]{false};

    SortPlanFixture() {
        value = makeIntConst();

        target.xpr.type = T_TargetEntry;
        target.expr = reinterpret_cast<Expr*>(&value);
        target.resno = 1;
        target.resjunk = false;

        sort.plan.type = T_Sort;
        sort.plan.targetlist = list_make1(&target);
        sort.numCols = 1;
        sort.sortColIdx = sortColIdx;
        sort.sortOperators = sortOperators;
        sort.collations = collations;
        sort.nullsFirst = nullsFirst;
    }
};

struct AggPlanFixture {
    Const value{};
    TargetEntry target{};
    Agg agg{};
    AttrNumber grpColIdx[1]{1};
    Oid grpOperators[1]{Int4EqualOperator};
    Oid grpCollations[1]{InvalidOid};

    AggPlanFixture() {
        value = makeIntConst();

        target.xpr.type = T_TargetEntry;
        target.expr = reinterpret_cast<Expr*>(&value);
        target.resno = 1;
        target.resjunk = false;

        agg.plan.type = T_Agg;
        agg.plan.targetlist = list_make1(&target);
        agg.numCols = 1;
        agg.grpColIdx = grpColIdx;
        agg.grpOperators = grpOperators;
        agg.grpCollations = grpCollations;
    }
};

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

PGX_TEST_FN(query_analyzer_rejects_boolean_test) {
    auto arg = makeBoolConst();
    auto booleanTest = BooleanTest{};
    booleanTest.xpr.type = T_BooleanTest;
    booleanTest.arg = reinterpret_cast<Expr*>(&arg);
    booleanTest.booltesttype = IS_TRUE;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&booleanTest));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_coalesce_expr) {
    auto nullableArg = makeIntConst();
    auto fallbackArg = makeIntConst();
    auto coalesce = CoalesceExpr{};
    coalesce.xpr.type = T_CoalesceExpr;
    coalesce.coalescetype = INT4OID;
    coalesce.coalescecollid = InvalidOid;
    coalesce.args = list_make2(&nullableArg, &fallbackArg);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&coalesce));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_scalar_array_op_expr) {
    auto lhs = makeIntConst();
    auto elem1 = makeIntConst();
    auto elem2 = makeIntConst();
    auto arrayExpr = ArrayExpr{};
    arrayExpr.xpr.type = T_ArrayExpr;
    arrayExpr.array_typeid = INT4ARRAYOID;
    arrayExpr.element_typeid = INT4OID;
    arrayExpr.elements = list_make2(&elem1, &elem2);

    auto scalarArray = ScalarArrayOpExpr{};
    scalarArray.xpr.type = T_ScalarArrayOpExpr;
    scalarArray.opno = Int4EqualOperator;
    scalarArray.opfuncid = InvalidOid;
    scalarArray.useOr = true;
    scalarArray.inputcollid = InvalidOid;
    scalarArray.args = list_make2(&lhs, &arrayExpr);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&scalarArray));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_scalar_array_ordering_operator) {
    auto lhs = makeIntConst();
    auto elem1 = makeIntConst();
    auto elem2 = makeIntConst();
    auto arrayExpr = ArrayExpr{};
    arrayExpr.xpr.type = T_ArrayExpr;
    arrayExpr.array_typeid = INT4ARRAYOID;
    arrayExpr.element_typeid = INT4OID;
    arrayExpr.elements = list_make2(&elem1, &elem2);

    auto scalarArray = ScalarArrayOpExpr{};
    scalarArray.xpr.type = T_ScalarArrayOpExpr;
    scalarArray.opno = Int4LessOperator;
    scalarArray.opfuncid = InvalidOid;
    scalarArray.useOr = true;
    scalarArray.inputcollid = InvalidOid;
    scalarArray.args = list_make2(&lhs, &arrayExpr);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&scalarArray));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_scalar_array_text_const_array) {
    auto lhs = makeTypedConst(TEXTOID);
    auto arrayConst = makeTypedConst(TEXTARRAYOID);
    auto scalarArray = ScalarArrayOpExpr{};
    scalarArray.xpr.type = T_ScalarArrayOpExpr;
    scalarArray.opno = TextEqualOperator;
    scalarArray.opfuncid = InvalidOid;
    scalarArray.useOr = true;
    scalarArray.inputcollid = DEFAULT_COLLATION_OID;
    scalarArray.args = list_make2(&lhs, &arrayConst);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&scalarArray));
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

PGX_TEST_FN(query_analyzer_rejects_invalid_operator_oid) {
    auto lhs = makeBoolConst();
    auto rhs = makeBoolConst();
    auto op = OpExpr{};
    op.xpr.type = T_OpExpr;
    op.opno = InvalidOid;
    op.opfuncid = InvalidOid;
    op.opresulttype = BOOLOID;
    op.inputcollid = InvalidOid;
    op.opcollid = InvalidOid;
    op.args = list_make2(&lhs, &rhs);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_exposed_bytea_value) {
    auto byteaValue = makeTypedConst(BYTEAOID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&byteaValue));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_bytea_typed_supported_aggregate) {
    auto argValue = makeTypedConst(INT8OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT8;
    aggregate.aggtype = BYTEAOID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(INT8OID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_bytea_typed_aggregate_without_argtypes) {
    auto argValue = makeTypedConst(INT8OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT8;
    aggregate.aggtype = BYTEAOID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = NIL;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_aggregate_collation) {
    auto argValue = makeTypedConst(TEXTOID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_MIN_TEXT;
    aggregate.aggtype = TEXTOID;
    aggregate.aggcollid = 999999;
    aggregate.inputcollid = 999999;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(TEXTOID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_collation);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_min_text_aggregate) {
    auto argValue = makeTypedConst(TEXTOID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_MIN_TEXT;
    aggregate.aggtype = TEXTOID;
    aggregate.aggcollid = DEFAULT_COLLATION_OID;
    aggregate.inputcollid = DEFAULT_COLLATION_OID;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(TEXTOID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_max_text_aggregate) {
    auto argValue = makeTypedConst(TEXTOID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_MAX_TEXT;
    aggregate.aggtype = TEXTOID;
    aggregate.aggcollid = DEFAULT_COLLATION_OID;
    aggregate.inputcollid = DEFAULT_COLLATION_OID;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(TEXTOID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_min_bpchar_aggregate) {
    auto argValue = makeTypedConst(BPCHAROID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_MIN_BPCHAR;
    aggregate.aggtype = BPCHAROID;
    aggregate.aggcollid = DEFAULT_COLLATION_OID;
    aggregate.inputcollid = DEFAULT_COLLATION_OID;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(BPCHAROID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_max_bpchar_aggregate) {
    auto argValue = makeTypedConst(BPCHAROID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_MAX_BPCHAR;
    aggregate.aggtype = BPCHAROID;
    aggregate.aggcollid = DEFAULT_COLLATION_OID;
    aggregate.inputcollid = DEFAULT_COLLATION_OID;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(BPCHAROID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_string_coerce_via_io) {
    auto arg = makeTypedConst(TEXTOID);
    auto coerce = CoerceViaIO{};
    coerce.xpr.type = T_CoerceViaIO;
    coerce.arg = reinterpret_cast<Expr*>(&arg);
    coerce.resulttype = VARCHAROID;
    coerce.resultcollid = DEFAULT_COLLATION_OID;
    coerce.coerceformat = COERCE_IMPLICIT_CAST;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&coerce));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_non_string_coerce_via_io) {
    auto arg = makeIntConst();
    auto coerce = CoerceViaIO{};
    coerce.xpr.type = T_CoerceViaIO;
    coerce.arg = reinterpret_cast<Expr*>(&arg);
    coerce.resulttype = TEXTOID;
    coerce.resultcollid = DEFAULT_COLLATION_OID;
    coerce.coerceformat = COERCE_EXPLICIT_CAST;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&coerce));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_varchar_operator_signature) {
    auto lhs = makeTypedConst(VARCHAROID);
    auto rhs = makeTypedConst(VARCHAROID);
    auto op = OpExpr{};
    op.xpr.type = T_OpExpr;
    op.opno = TextEqualOperator;
    op.opfuncid = F_TEXTEQ;
    op.opresulttype = BOOLOID;
    op.inputcollid = DEFAULT_COLLATION_OID;
    op.opcollid = InvalidOid;
    op.args = list_make2(&lhs, &rhs);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_invalid_sort_operator) {
    SortPlanFixture fixture;
    fixture.sortOperators[0] = InvalidOid;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_sort_collation) {
    SortPlanFixture fixture;
    fixture.collations[0] = 999999;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_collation);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_sort_operator_metadata) {
    SortPlanFixture fixture;
    fixture.sort.sortOperators = nullptr;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_sort_column_metadata) {
    SortPlanFixture fixture;
    fixture.sort.sortColIdx = nullptr;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_sort_nulls_first_metadata) {
    SortPlanFixture fixture;
    fixture.sort.nullsFirst = nullptr;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_non_ordering_sort_operator) {
    SortPlanFixture fixture;
    fixture.sortOperators[0] = Int4EqualOperator;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_non_strict_sort_operator) {
    SortPlanFixture fixture;
    fixture.sortOperators[0] = Int4LessEqualOperator;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_explicit_ascending_nulls_first_sort) {
    SortPlanFixture fixture;
    fixture.nullsFirst[0] = true;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_explicit_descending_nulls_last_sort) {
    SortPlanFixture fixture;
    fixture.sortOperators[0] = Int4GreaterOperator;
    fixture.nullsFirst[0] = false;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_agg_group_column_metadata) {
    AggPlanFixture fixture;
    fixture.agg.grpColIdx = nullptr;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_agg_group_operator_metadata) {
    AggPlanFixture fixture;
    fixture.agg.grpOperators = nullptr;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_missing_agg_group_collation_metadata) {
    AggPlanFixture fixture;
    fixture.agg.grpCollations = nullptr;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::missing_metadata);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_non_equality_agg_group_operator) {
    AggPlanFixture fixture;
    fixture.grpOperators[0] = Int4LessOperator;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_agg_group_collation) {
    AggPlanFixture fixture;
    fixture.grpCollations[0] = 999999;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_collation);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_agg_grouping_sets) {
    AggPlanFixture fixture;
    fixture.agg.groupingSets = list_make1_int(1);

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_agg_chain) {
    AggPlanFixture fixture;
    fixture.agg.chain = list_make1_int(1);

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_aggregate_filter) {
    auto filter = makeBoolConst();

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_COUNT_ANY;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.aggfilter = reinterpret_cast<Expr*>(&filter);
    aggregate.aggargtypes = NIL;
    aggregate.args = NIL;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_aggregate_ordering) {
    auto argValue = makeTypedConst(FLOAT8OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;
    argTarget.ressortgroupref = 1;

    auto sortClause = SortGroupClause{};
    sortClause.tleSortGroupRef = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_FLOAT8;
    aggregate.aggtype = FLOAT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(FLOAT8OID);
    aggregate.aggorder = list_make1(&sortClause);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_seq_scan_without_sort_metadata) {
    auto value = makeIntConst();
    auto target = TargetEntry{};
    target.xpr.type = T_TargetEntry;
    target.expr = reinterpret_cast<Expr*>(&value);
    target.resno = 1;
    target.resjunk = false;

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = list_make1(&target);
    scan.scan.scanrelid = 1;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(result.isSupported());
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
