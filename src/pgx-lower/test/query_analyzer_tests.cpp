extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/namespace.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/nodeFuncs.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/timestamp.h"
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
constexpr int32_t Numeric12_2Typmod = ((12 << 16) | 2) + VARHDRSZ;
constexpr int32_t Varchar16Typmod = 16 + VARHDRSZ;
constexpr int32_t Bpchar4Typmod = 4 + VARHDRSZ;
constexpr int32_t IntervalYearToMonthTypmod = 0x7FFF0004;

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

auto makeTypedConst(Oid typeOid, int32_t typmod = -1, Oid collation = InvalidOid, bool isNull = false) -> Const {
    auto value = Const{};
    value.xpr.type = T_Const;
    value.consttype = typeOid;
    value.consttypmod = typmod;
    value.constcollid = collation;
    if (!OidIsValid(value.constcollid) && OidIsValid(typeOid)
        && (typeOid == TEXTOID || typeOid == VARCHAROID || typeOid == BPCHAROID))
    {
        value.constcollid = DEFAULT_COLLATION_OID;
    }
    value.constisnull = isNull;
    value.constbyval = false;
    value.constlen = -1;
    value.constvalue = Datum{0};
    return value;
}

auto makeTextConst(const char* text) -> Const {
    auto value = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    value.constvalue = CStringGetTextDatum(text);
    return value;
}

auto makeIntervalConst(int64_t time, int32_t day, int32_t month) -> Const {
    auto* interval = static_cast<Interval*>(palloc0(sizeof(Interval)));
    interval->time = time;
    interval->day = day;
    interval->month = month;

    auto value = makeTypedConst(INTERVALOID);
    value.constlen = sizeof(Interval);
    value.constvalue = IntervalPGetDatum(interval);
    return value;
}

void requireUnsupportedTemporalConst(Oid typeOid) {
    auto value = makeTypedConst(typeOid);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&value));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);
    REQUIRE(result.humanSummary().find("unsupported PostgreSQL type OID") != std::string::npos);
}

auto makeIntVar(AttrNumber attno) -> Var {
    auto value = Var{};
    value.xpr.type = T_Var;
    value.varno = OUTER_VAR;
    value.varattno = attno;
    value.vartype = INT4OID;
    value.vartypmod = -1;
    value.varcollid = InvalidOid;
    value.varlevelsup = 0;
    return value;
}

auto makeTypedVar(Oid typeOid, AttrNumber attno) -> Var {
    auto value = Var{};
    value.xpr.type = T_Var;
    value.varno = OUTER_VAR;
    value.varattno = attno;
    value.vartype = typeOid;
    value.vartypmod = -1;
    value.varcollid = InvalidOid;
    value.varlevelsup = 0;
    return value;
}

auto makeTypedVar(Oid typeOid, AttrNumber attno, int32_t typmod, Oid collation) -> Var {
    auto value = makeTypedVar(typeOid, attno);
    value.vartypmod = typmod;
    value.varcollid = collation;
    return value;
}

auto makeBinaryOperatorExpr(const char* name, Oid resultType, Node* lhs, Node* rhs) -> OpExpr {
    const auto lhsType = exprType(lhs);
    const auto rhsType = exprType(rhs);
    const Oid opOid = OpernameGetOprid(list_make1(makeString(const_cast<char*>(name))), lhsType, rhsType);

    auto op = OpExpr{};
    op.xpr.type = T_OpExpr;
    op.opno = opOid;
    op.opfuncid = get_opcode(opOid);
    op.opresulttype = resultType;
    op.inputcollid = InvalidOid;
    op.opcollid = InvalidOid;
    op.args = list_make2(lhs, rhs);
    return op;
}

auto makeRelabel(Expr* arg, Oid resultType, int32_t resultTypmod, Oid resultCollation) -> RelabelType {
    auto relabel = RelabelType{};
    relabel.xpr.type = T_RelabelType;
    relabel.arg = arg;
    relabel.resulttype = resultType;
    relabel.resulttypmod = resultTypmod;
    relabel.resultcollid = resultCollation;
    relabel.relabelformat = COERCE_EXPLICIT_CAST;
    return relabel;
}

auto makeFunctionExpr(Oid functionOid, Oid resultType, Oid inputCollation, Oid resultCollation, List* args) -> FuncExpr {
    auto func = FuncExpr{};
    func.xpr.type = T_FuncExpr;
    func.funcid = functionOid;
    func.funcresulttype = resultType;
    func.inputcollid = inputCollation;
    func.funccollid = resultCollation;
    func.args = args;
    return func;
}

auto makeOutputTarget(Expr* expr, AttrNumber resno) -> TargetEntry {
    auto target = TargetEntry{};
    target.xpr.type = T_TargetEntry;
    target.expr = expr;
    target.resno = resno;
    target.resjunk = false;
    return target;
}

struct IntervalAggregateFixture {
    Const value{};
    TargetEntry argument{};
    Aggref aggregate{};

    explicit IntervalAggregateFixture(Oid aggregateFunctionOid) {
        value = makeIntervalConst(0, 1, 1);

        argument.xpr.type = T_TargetEntry;
        argument.expr = reinterpret_cast<Expr*>(&value);
        argument.resno = 1;

        aggregate.xpr.type = T_Aggref;
        aggregate.aggfnoid = aggregateFunctionOid;
        aggregate.aggtype = INTERVALOID;
        aggregate.aggcollid = InvalidOid;
        aggregate.inputcollid = InvalidOid;
        aggregate.args = list_make1(&argument);
        aggregate.aggargtypes = list_make1_oid(INTERVALOID);
    }
};

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

PGX_TEST_FN(query_analyzer_string_type_boundary) {
    auto text = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto varchar = makeTypedConst(VARCHAROID, 16, DEFAULT_COLLATION_OID);
    auto bpchar = makeTypedConst(BPCHAROID, 8, DEFAULT_COLLATION_OID);
    auto bytea = makeTypedConst(BYTEAOID);
    auto internalChar = makeTypedConst(CHAROID);
    auto nameValue = makeTypedConst(NAMEOID);
    auto cstringValue = makeTypedConst(CSTRINGOID);

    REQUIRE(pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&text)).isSupported());
    REQUIRE(pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&varchar)).isSupported());
    REQUIRE(pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&bpchar)).isSupported());

    const auto byteaResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&bytea));
    REQUIRE(!byteaResult.isSupported());
    REQUIRE(byteaResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);

    const auto charResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&internalChar));
    REQUIRE(!charResult.isSupported());
    REQUIRE(charResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);

    const auto nameResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&nameValue));
    REQUIRE(!nameResult.isSupported());
    REQUIRE(nameResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);

    const auto cstringResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&cstringValue));
    REQUIRE(!cstringResult.isSupported());
    REQUIRE(cstringResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_temporal_type_oids) {
    requireUnsupportedTemporalConst(TIMEOID);
    requireUnsupportedTemporalConst(TIMETZOID);
    requireUnsupportedTemporalConst(TIMESTAMPTZOID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_interval_projection) {
    auto interval = makeIntervalConst(123456789, -7, 14);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&interval));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_date_int4_arithmetic) {
    auto date = makeTypedConst(DATEOID);
    auto int4 = makeTypedConst(INT4OID);
    auto int8 = makeTypedConst(INT8OID);

    auto datePlusInt4 = makeBinaryOperatorExpr("+", DATEOID, reinterpret_cast<Node*>(&date),
                                               reinterpret_cast<Node*>(&int4));
    REQUIRE(OidIsValid(datePlusInt4.opno));
    REQUIRE(OidIsValid(datePlusInt4.opfuncid));
    const auto plusInt4Result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&datePlusInt4));
    REQUIRE(plusInt4Result.isSupported());

    auto dateMinusInt4 = makeBinaryOperatorExpr("-", DATEOID, reinterpret_cast<Node*>(&date),
                                                reinterpret_cast<Node*>(&int4));
    REQUIRE(OidIsValid(dateMinusInt4.opno));
    REQUIRE(OidIsValid(dateMinusInt4.opfuncid));
    const auto minusInt4Result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&dateMinusInt4));
    REQUIRE(minusInt4Result.isSupported());

    auto datePlusInt8 = makeBinaryOperatorExpr("+", DATEOID, reinterpret_cast<Node*>(&date),
                                               reinterpret_cast<Node*>(&int8));
    const auto plusInt8Result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&datePlusInt8));
    REQUIRE(!plusInt8Result.isSupported());
    REQUIRE(plusInt8Result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);

    auto dateMinusInt8 = makeBinaryOperatorExpr("-", DATEOID, reinterpret_cast<Node*>(&date),
                                                reinterpret_cast<Node*>(&int8));
    const auto minusInt8Result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&dateMinusInt8));
    REQUIRE(!minusInt8Result.isSupported());
    REQUIRE(minusInt8Result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_extract_from_date) {
    for (const char* fieldName : {"year", "month", "day"}) {
        auto field = makeTextConst(fieldName);
        auto date = makeTypedConst(DATEOID);
        auto extract = FuncExpr{};
        extract.xpr.type = T_FuncExpr;
        extract.funcid = F_EXTRACT_TEXT_DATE;
        extract.funcresulttype = NUMERICOID;
        extract.inputcollid = DEFAULT_COLLATION_OID;
        extract.funccollid = InvalidOid;
        extract.args = list_make2(&field, &date);

        const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&extract));
        REQUIRE(result.isSupported());
    }
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_extract_field_from_date) {
    auto field = makeTextConst("quarter");
    auto date = makeTypedConst(DATEOID);
    auto extract = FuncExpr{};
    extract.xpr.type = T_FuncExpr;
    extract.funcid = F_EXTRACT_TEXT_DATE;
    extract.funcresulttype = NUMERICOID;
    extract.inputcollid = DEFAULT_COLLATION_OID;
    extract.funccollid = InvalidOid;
    extract.args = list_make2(&field, &date);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&extract));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_extract_from_timestamp) {
    auto field = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto timestamp = makeTypedConst(TIMESTAMPOID);
    auto extract = FuncExpr{};
    extract.xpr.type = T_FuncExpr;
    extract.funcid = F_EXTRACT_TEXT_TIMESTAMP;
    extract.funcresulttype = NUMERICOID;
    extract.inputcollid = DEFAULT_COLLATION_OID;
    extract.funccollid = InvalidOid;
    extract.args = list_make2(&field, &timestamp);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&extract));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_timestamp_without_time_zone_comparison) {
    auto lhs = makeTypedConst(TIMESTAMPOID);
    auto rhs = makeTypedConst(TIMESTAMPOID);
    auto op = makeBinaryOperatorExpr(">", BOOLOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));

    REQUIRE(OidIsValid(op.opno));
    REQUIRE(OidIsValid(op.opfuncid));
    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_unsupported_temporal_comparisons) {
    for (const auto typeOid : {TIMEOID, TIMETZOID, TIMESTAMPTZOID}) {
        auto lhs = makeTypedConst(typeOid);
        auto rhs = makeTypedConst(typeOid);
        auto op = makeBinaryOperatorExpr(">", BOOLOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));

        REQUIRE(OidIsValid(op.opno));
        REQUIRE(OidIsValid(op.opfuncid));
        const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
        REQUIRE(!result.isSupported());
        const bool expectedReason = result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type
                                    || result.primaryReason().kind
                                           == pgx_lower::UnsupportedReasonKind::unsupported_operator;
        REQUIRE(expectedReason);
    }
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_day_interval_date_arithmetic) {
    auto date = makeTypedConst(DATEOID);
    auto dayInterval = makeIntervalConst(0, 90, 0);

    auto datePlusInterval = makeBinaryOperatorExpr("+", TIMESTAMPOID, reinterpret_cast<Node*>(&date),
                                                   reinterpret_cast<Node*>(&dayInterval));
    const auto plusResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&datePlusInterval));
    REQUIRE(plusResult.isSupported());

    auto dateMinusInterval = makeBinaryOperatorExpr("-", TIMESTAMPOID, reinterpret_cast<Node*>(&date),
                                                    reinterpret_cast<Node*>(&dayInterval));
    const auto minusResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&dateMinusInterval));
    REQUIRE(minusResult.isSupported());

    auto intervalPlusDate = makeBinaryOperatorExpr("+", TIMESTAMPOID, reinterpret_cast<Node*>(&dayInterval),
                                                   reinterpret_cast<Node*>(&date));
    const auto commutedResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(
        reinterpret_cast<Node*>(&intervalPlusDate));
    REQUIRE(commutedResult.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_month_interval_date_arithmetic) {
    auto date = makeTypedConst(DATEOID);
    auto monthInterval = makeIntervalConst(0, 0, 1);
    auto op = makeBinaryOperatorExpr("+", TIMESTAMPOID, reinterpret_cast<Node*>(&date),
                                     reinterpret_cast<Node*>(&monthInterval));

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    REQUIRE(result.humanSummary().find("interval month") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_nonconstant_date_interval_arithmetic) {
    auto date = makeTypedConst(DATEOID);
    auto intervalVar = makeTypedVar(INTERVALOID, 1);
    auto op = makeBinaryOperatorExpr("+", TIMESTAMPOID, reinterpret_cast<Node*>(&date),
                                     reinterpret_cast<Node*>(&intervalVar));

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    REQUIRE(result.humanSummary().find("unsupported interval semantics") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_interval_comparison) {
    auto lhs = makeIntervalConst(0, 0, 1);
    auto rhs = makeIntervalConst(0, 1, 0);

    auto equality = makeBinaryOperatorExpr("=", BOOLOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));
    const auto equalityResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&equality));
    REQUIRE(!equalityResult.isSupported());
    REQUIRE(equalityResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);

    auto ordering = makeBinaryOperatorExpr("<", BOOLOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));
    const auto orderingResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&ordering));
    REQUIRE(!orderingResult.isSupported());
    REQUIRE(orderingResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    REQUIRE(orderingResult.humanSummary().find("unsupported interval semantics") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_timestamp_interval_arithmetic) {
    auto timestamp = makeTypedConst(TIMESTAMPOID);
    auto dayInterval = makeIntervalConst(0, 1, 0);
    auto op = makeBinaryOperatorExpr("+", TIMESTAMPOID, reinterpret_cast<Node*>(&timestamp),
                                     reinterpret_cast<Node*>(&dayInterval));

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    REQUIRE(result.humanSummary().find("unsupported interval semantics") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_timestamp_minus_timestamp_interval) {
    auto lhs = makeTypedConst(TIMESTAMPOID);
    auto rhs = makeTypedConst(TIMESTAMPOID);
    auto op = makeBinaryOperatorExpr("-", INTERVALOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    REQUIRE(result.humanSummary().find("unsupported interval semantics") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_interval_interval_arithmetic) {
    auto lhs = makeIntervalConst(0, 0, 1);
    auto rhs = makeIntervalConst(0, 1, 0);

    auto plus = makeBinaryOperatorExpr("+", INTERVALOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));
    const auto plusResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&plus));
    REQUIRE(!plusResult.isSupported());
    REQUIRE(plusResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);

    auto minus = makeBinaryOperatorExpr("-", INTERVALOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));
    const auto minusResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&minus));
    REQUIRE(!minusResult.isSupported());
    REQUIRE(minusResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);
    REQUIRE(minusResult.humanSummary().find("unsupported interval semantics") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_interval_aggregates) {
    for (const auto aggregateOid : {F_SUM_INTERVAL, F_AVG_INTERVAL, F_MIN_INTERVAL, F_MAX_INTERVAL}) {
        IntervalAggregateFixture fixture(aggregateOid);
        const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&fixture.aggregate));
        REQUIRE(!result.isSupported());
        REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
        REQUIRE(result.humanSummary().find("unsupported interval aggregate") != std::string::npos);
    }
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_count_interval) {
    auto value = makeIntervalConst(0, 1, 1);

    auto argument = TargetEntry{};
    argument.xpr.type = T_TargetEntry;
    argument.expr = reinterpret_cast<Expr*>(&value);
    argument.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_COUNT_ANY;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argument);
    aggregate.aggargtypes = list_make1_oid(INTERVALOID);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_string_operator_boundary) {
    auto textLhs = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto textRhs = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto textEq = OpExpr{};
    textEq.xpr.type = T_OpExpr;
    textEq.opno = TextEqualOperator;
    textEq.opfuncid = F_TEXTEQ;
    textEq.opresulttype = BOOLOID;
    textEq.inputcollid = DEFAULT_COLLATION_OID;
    textEq.opcollid = InvalidOid;
    textEq.args = list_make2(&textLhs, &textRhs);

    const auto textEqResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&textEq));
    REQUIRE(textEqResult.isSupported());

    auto textLt = makeBinaryOperatorExpr("<", BOOLOID, reinterpret_cast<Node*>(&textLhs),
                                         reinterpret_cast<Node*>(&textRhs));
    textLt.inputcollid = DEFAULT_COLLATION_OID;
    const auto textLtResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&textLt));
    REQUIRE(textLtResult.isSupported());

    const Oid bpcharEqOid = OpernameGetOprid(list_make1(makeString(const_cast<char*>("="))), BPCHAROID, BPCHAROID);
    auto bpcharLhs = makeTypedConst(BPCHAROID, 8, DEFAULT_COLLATION_OID);
    auto bpcharRhs = makeTypedConst(BPCHAROID, 8, DEFAULT_COLLATION_OID);
    auto bpcharEq = OpExpr{};
    bpcharEq.xpr.type = T_OpExpr;
    bpcharEq.opno = bpcharEqOid;
    bpcharEq.opfuncid = get_opcode(bpcharEqOid);
    bpcharEq.opresulttype = BOOLOID;
    bpcharEq.inputcollid = DEFAULT_COLLATION_OID;
    bpcharEq.opcollid = InvalidOid;
    bpcharEq.args = list_make2(&bpcharLhs, &bpcharRhs);

    const auto bpcharEqResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&bpcharEq));
    REQUIRE(bpcharEqResult.isSupported());

    auto bpcharLt = makeBinaryOperatorExpr("<", BOOLOID, reinterpret_cast<Node*>(&bpcharLhs),
                                           reinterpret_cast<Node*>(&bpcharRhs));
    bpcharLt.inputcollid = DEFAULT_COLLATION_OID;
    const auto bpcharLtResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&bpcharLt));
    REQUIRE(bpcharLtResult.isSupported());

    auto varcharLhs = makeTypedConst(VARCHAROID, 16, DEFAULT_COLLATION_OID);
    auto varcharRhs = makeTypedConst(VARCHAROID, 16, DEFAULT_COLLATION_OID);
    auto varcharEq = OpExpr{};
    varcharEq.xpr.type = T_OpExpr;
    varcharEq.opno = TextEqualOperator;
    varcharEq.opfuncid = F_TEXTEQ;
    varcharEq.opresulttype = BOOLOID;
    varcharEq.inputcollid = DEFAULT_COLLATION_OID;
    varcharEq.opcollid = InvalidOid;
    varcharEq.args = list_make2(&varcharLhs, &varcharRhs);

    const auto varcharEqResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&varcharEq));
    REQUIRE(!varcharEqResult.isSupported());
    REQUIRE(varcharEqResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_operator);

    auto likePattern = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto textLike = OpExpr{};
    textLike.xpr.type = T_OpExpr;
    textLike.opno = OID_TEXT_LIKE_OP;
    textLike.opfuncid = F_TEXTLIKE;
    textLike.opresulttype = BOOLOID;
    textLike.inputcollid = DEFAULT_COLLATION_OID;
    textLike.opcollid = InvalidOid;
    textLike.args = list_make2(&textLhs, &likePattern);

    const auto textLikeResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&textLike));
    REQUIRE(textLikeResult.isSupported());

    const Oid textNotLikeOid = OpernameGetOprid(list_make1(makeString(const_cast<char*>("!~~"))), TEXTOID, TEXTOID);
    auto textNotLike = OpExpr{};
    textNotLike.xpr.type = T_OpExpr;
    textNotLike.opno = textNotLikeOid;
    textNotLike.opfuncid = F_TEXTNLIKE;
    textNotLike.opresulttype = BOOLOID;
    textNotLike.inputcollid = DEFAULT_COLLATION_OID;
    textNotLike.opcollid = InvalidOid;
    textNotLike.args = list_make2(&textLhs, &likePattern);

    const auto textNotLikeResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&textNotLike));
    REQUIRE(textNotLikeResult.isSupported());

    auto bpcharLike = OpExpr{};
    bpcharLike.xpr.type = T_OpExpr;
    bpcharLike.opno = OID_BPCHAR_LIKE_OP;
    bpcharLike.opfuncid = F_BPCHARLIKE;
    bpcharLike.opresulttype = BOOLOID;
    bpcharLike.inputcollid = DEFAULT_COLLATION_OID;
    bpcharLike.opcollid = InvalidOid;
    bpcharLike.args = list_make2(&bpcharLhs, &likePattern);

    const auto bpcharLikeResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&bpcharLike));
    REQUIRE(bpcharLikeResult.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_varchar_relabel_string_operator_boundary) {
    auto varcharLhs = makeTypedConst(VARCHAROID, 16, DEFAULT_COLLATION_OID);
    auto varcharRhs = makeTypedConst(VARCHAROID, 16, DEFAULT_COLLATION_OID);
    auto textPattern = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto lhsAsText = makeRelabel(reinterpret_cast<Expr*>(&varcharLhs), TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto rhsAsText = makeRelabel(reinterpret_cast<Expr*>(&varcharRhs), TEXTOID, -1, DEFAULT_COLLATION_OID);

    auto eq = makeBinaryOperatorExpr("=", BOOLOID, reinterpret_cast<Node*>(&lhsAsText),
                                     reinterpret_cast<Node*>(&rhsAsText));
    eq.inputcollid = DEFAULT_COLLATION_OID;
    const auto eqResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&eq));
    REQUIRE(eqResult.isSupported());

    auto lt = makeBinaryOperatorExpr("<", BOOLOID, reinterpret_cast<Node*>(&lhsAsText),
                                     reinterpret_cast<Node*>(&rhsAsText));
    lt.inputcollid = DEFAULT_COLLATION_OID;
    const auto ltResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&lt));
    REQUIRE(ltResult.isSupported());

    auto like = makeBinaryOperatorExpr("~~", BOOLOID, reinterpret_cast<Node*>(&lhsAsText),
                                       reinterpret_cast<Node*>(&textPattern));
    like.inputcollid = DEFAULT_COLLATION_OID;
    const auto likeResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&like));
    REQUIRE(likeResult.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_non_default_string_operator_collation) {
    auto lhs = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto rhs = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto op = makeBinaryOperatorExpr("<", BOOLOID, reinterpret_cast<Node*>(&lhs), reinterpret_cast<Node*>(&rhs));
    op.inputcollid = C_COLLATION_OID;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&op));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_collation);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_string_function_boundary) {
    auto text = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto varchar = makeTypedConst(VARCHAROID, 16, DEFAULT_COLLATION_OID);
    auto varcharAsText = makeRelabel(reinterpret_cast<Expr*>(&varchar), TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto bpchar = makeTypedConst(BPCHAROID, 8, DEFAULT_COLLATION_OID);
    auto bpcharAsText = makeRelabel(reinterpret_cast<Expr*>(&bpchar), TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto start = makeTypedConst(INT4OID);
    auto length = makeTypedConst(INT4OID);

    auto upperText = makeFunctionExpr(F_UPPER_TEXT, TEXTOID, DEFAULT_COLLATION_OID, DEFAULT_COLLATION_OID,
                                      list_make1(&text));
    const auto upperTextResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&upperText));
    REQUIRE(upperTextResult.isSupported());

    auto lowerVarchar = makeFunctionExpr(F_LOWER_TEXT, TEXTOID, DEFAULT_COLLATION_OID, DEFAULT_COLLATION_OID,
                                         list_make1(&varcharAsText));
    const auto lowerVarcharResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(
        reinterpret_cast<Node*>(&lowerVarchar));
    REQUIRE(lowerVarcharResult.isSupported());

    auto substringText = makeFunctionExpr(F_SUBSTRING_TEXT_INT4_INT4, TEXTOID, DEFAULT_COLLATION_OID,
                                          DEFAULT_COLLATION_OID, list_make3(&text, &start, &length));
    const auto substringTextResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(
        reinterpret_cast<Node*>(&substringText));
    REQUIRE(substringTextResult.isSupported());

    auto substringTextWithoutLength = makeFunctionExpr(F_SUBSTRING_TEXT_INT4, TEXTOID, DEFAULT_COLLATION_OID,
                                                       DEFAULT_COLLATION_OID, list_make2(&text, &start));
    const auto substringTextWithoutLengthResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(
        reinterpret_cast<Node*>(&substringTextWithoutLength));
    REQUIRE(substringTextWithoutLengthResult.isSupported());

    auto substringVarchar = makeFunctionExpr(F_SUBSTRING_TEXT_INT4_INT4, TEXTOID, DEFAULT_COLLATION_OID,
                                             DEFAULT_COLLATION_OID, list_make3(&varcharAsText, &start, &length));
    const auto substringVarcharResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(
        reinterpret_cast<Node*>(&substringVarchar));
    REQUIRE(substringVarcharResult.isSupported());

    auto upperBpchar = makeFunctionExpr(F_UPPER_TEXT, TEXTOID, DEFAULT_COLLATION_OID, DEFAULT_COLLATION_OID,
                                        list_make1(&bpcharAsText));
    const auto upperBpcharResult = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&upperBpchar));
    REQUIRE(!upperBpcharResult.isSupported());
    REQUIRE(upperBpcharResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_string_cast_boundary) {
    auto arg = makeTypedConst(TEXTOID, -1, DEFAULT_COLLATION_OID);
    auto coerce = CoerceViaIO{};
    coerce.xpr.type = T_CoerceViaIO;
    coerce.arg = reinterpret_cast<Expr*>(&arg);
    coerce.resulttype = VARCHAROID;
    coerce.resultcollid = DEFAULT_COLLATION_OID;
    coerce.coerceformat = COERCE_IMPLICIT_CAST;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&coerce));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_relabel_string_cast_boundary) {
    auto arg = makeTypedConst(VARCHAROID, 12, DEFAULT_COLLATION_OID);
    auto relabel = RelabelType{};
    relabel.xpr.type = T_RelabelType;
    relabel.arg = reinterpret_cast<Expr*>(&arg);
    relabel.resulttype = TEXTOID;
    relabel.resulttypmod = -1;
    relabel.resultcollid = DEFAULT_COLLATION_OID;
    relabel.relabelformat = COERCE_EXPLICIT_CAST;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&relabel));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_function);
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

PGX_TEST_FN(query_analyzer_accepts_bpchar_sort_operator) {
    auto bpchar = makeTypedConst(BPCHAROID, 8, DEFAULT_COLLATION_OID);
    SortPlanFixture fixture;
    fixture.value = bpchar;
    fixture.target.expr = reinterpret_cast<Expr*>(&fixture.value);
    fixture.sortOperators[0] = OpernameGetOprid(list_make1(makeString(const_cast<char*>("<"))), BPCHAROID, BPCHAROID);
    fixture.collations[0] = DEFAULT_COLLATION_OID;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.sort));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_resjunk_sort_key_metadata) {
    auto outputValue = makeIntVar(1);
    auto sortValue = makeIntVar(2);

    auto outputTarget = TargetEntry{};
    outputTarget.xpr.type = T_TargetEntry;
    outputTarget.expr = reinterpret_cast<Expr*>(&outputValue);
    outputTarget.resno = 1;
    outputTarget.resjunk = false;

    auto sortTarget = TargetEntry{};
    sortTarget.xpr.type = T_TargetEntry;
    sortTarget.expr = reinterpret_cast<Expr*>(&sortValue);
    sortTarget.resno = 2;
    sortTarget.resjunk = true;

    auto sort = Sort{};
    AttrNumber sortColIdx[1]{2};
    Oid sortOperators[1]{Int4LessOperator};
    Oid sortCollations[1]{InvalidOid};
    bool nullsFirst[1]{false};
    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    sort.plan.type = T_Sort;
    sort.plan.targetlist = list_make2(&outputTarget, &sortTarget);
    sort.plan.lefttree = reinterpret_cast<Plan*>(&scan);
    sort.numCols = 1;
    sort.sortColIdx = sortColIdx;
    sort.sortOperators = sortOperators;
    sort.collations = sortCollations;
    sort.nullsFirst = nullsFirst;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&sort));
    REQUIRE(result.isSupported());
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

PGX_TEST_FN(query_analyzer_accepts_bpchar_grouping_operator) {
    auto bpchar = makeTypedConst(BPCHAROID, 8, DEFAULT_COLLATION_OID);
    AggPlanFixture fixture;
    fixture.value = bpchar;
    fixture.target.expr = reinterpret_cast<Expr*>(&fixture.value);
    fixture.grpOperators[0] = OpernameGetOprid(list_make1(makeString(const_cast<char*>("="))), BPCHAROID, BPCHAROID);
    fixture.grpCollations[0] = DEFAULT_COLLATION_OID;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(result.isSupported());
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

PGX_TEST_FN(query_analyzer_rejects_string_agg_group_operator_for_child_target_type) {
    auto aggValue = makeTypedConst(INT4OID);
    auto aggTarget = TargetEntry{};
    aggTarget.xpr.type = T_TargetEntry;
    aggTarget.expr = reinterpret_cast<Expr*>(&aggValue);
    aggTarget.resno = 1;
    aggTarget.resjunk = false;

    auto childValue = makeTypedConst(BPCHAROID);
    auto childTarget = TargetEntry{};
    childTarget.xpr.type = T_TargetEntry;
    childTarget.expr = reinterpret_cast<Expr*>(&childValue);
    childTarget.resno = 1;
    childTarget.resjunk = false;

    auto childScan = SeqScan{};
    childScan.scan.plan.type = T_SeqScan;
    childScan.scan.plan.targetlist = list_make1(&childTarget);
    childScan.scan.scanrelid = 1;

    auto agg = Agg{};
    AttrNumber grpColIdx[1]{1};
    Oid grpOperators[1]{BpcharEqualOperator};
    Oid grpCollations[1]{InvalidOid};
    agg.plan.type = T_Agg;
    agg.plan.targetlist = list_make1(&aggTarget);
    agg.plan.lefttree = reinterpret_cast<Plan*>(&childScan);
    agg.numCols = 1;
    agg.grpColIdx = grpColIdx;
    agg.grpOperators = grpOperators;
    agg.grpCollations = grpCollations;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&agg));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_sorted_agg_over_join_input) {
    auto groupValue = makeIntVar(1);
    auto aggregateInput = makeIntVar(2);
    auto dependentValue = makeIntVar(3);

    auto groupTarget = TargetEntry{};
    groupTarget.xpr.type = T_TargetEntry;
    groupTarget.expr = reinterpret_cast<Expr*>(&groupValue);
    groupTarget.resno = 1;
    groupTarget.resjunk = false;

    auto aggregateInputTarget = TargetEntry{};
    aggregateInputTarget.xpr.type = T_TargetEntry;
    aggregateInputTarget.expr = reinterpret_cast<Expr*>(&aggregateInput);
    aggregateInputTarget.resno = 2;
    aggregateInputTarget.resjunk = false;

    auto dependentTarget = TargetEntry{};
    dependentTarget.xpr.type = T_TargetEntry;
    dependentTarget.expr = reinterpret_cast<Expr*>(&dependentValue);
    dependentTarget.resno = 3;
    dependentTarget.resjunk = false;

    auto aggregateArgTarget = TargetEntry{};
    aggregateArgTarget.xpr.type = T_TargetEntry;
    aggregateArgTarget.expr = reinterpret_cast<Expr*>(&aggregateInput);
    aggregateArgTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT4;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&aggregateArgTarget);
    aggregate.aggargtypes = list_make1_oid(INT4OID);
    aggregate.aggno = 1;

    auto aggregateTarget = TargetEntry{};
    aggregateTarget.xpr.type = T_TargetEntry;
    aggregateTarget.expr = reinterpret_cast<Expr*>(&aggregate);
    aggregateTarget.resno = 2;
    aggregateTarget.resjunk = false;

    auto leftScan = SeqScan{};
    leftScan.scan.plan.type = T_SeqScan;
    leftScan.scan.plan.targetlist = list_make3(&groupTarget, &aggregateInputTarget, &dependentTarget);

    auto rightScan = SeqScan{};
    rightScan.scan.plan.type = T_SeqScan;
    rightScan.scan.plan.targetlist = list_make3(&groupTarget, &aggregateInputTarget, &dependentTarget);

    auto join = NestLoop{};
    join.join.plan.type = T_NestLoop;
    join.join.plan.targetlist = list_make3(&groupTarget, &aggregateInputTarget, &dependentTarget);
    join.join.plan.lefttree = reinterpret_cast<Plan*>(&leftScan);
    join.join.plan.righttree = reinterpret_cast<Plan*>(&rightScan);

    auto sort = Sort{};
    AttrNumber sortColIdx[1]{1};
    Oid sortOperators[1]{Int4LessOperator};
    Oid sortCollations[1]{InvalidOid};
    bool nullsFirst[1]{false};
    sort.plan.type = T_Sort;
    sort.plan.targetlist = list_make3(&groupTarget, &aggregateInputTarget, &dependentTarget);
    sort.plan.lefttree = reinterpret_cast<Plan*>(&join);
    sort.numCols = 1;
    sort.sortColIdx = sortColIdx;
    sort.sortOperators = sortOperators;
    sort.collations = sortCollations;
    sort.nullsFirst = nullsFirst;

    auto agg = Agg{};
    AttrNumber grpColIdx[1]{1};
    Oid grpOperators[1]{Int4EqualOperator};
    Oid grpCollations[1]{InvalidOid};
    agg.plan.type = T_Agg;
    agg.plan.targetlist = list_make3(&groupTarget, &aggregateTarget, &dependentTarget);
    agg.plan.lefttree = reinterpret_cast<Plan*>(&sort);
    agg.aggstrategy = AGG_SORTED;
    agg.numCols = 1;
    agg.grpColIdx = grpColIdx;
    agg.grpOperators = grpOperators;
    agg.grpCollations = grpCollations;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_plan_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_sorted_agg_over_join_when_targets_are_grouped_or_aggregated) {
    auto groupValue = makeIntVar(1);
    auto aggregateInput = makeIntVar(2);

    auto groupTarget = TargetEntry{};
    groupTarget.xpr.type = T_TargetEntry;
    groupTarget.expr = reinterpret_cast<Expr*>(&groupValue);
    groupTarget.resno = 1;
    groupTarget.resjunk = false;

    auto aggregateInputTarget = TargetEntry{};
    aggregateInputTarget.xpr.type = T_TargetEntry;
    aggregateInputTarget.expr = reinterpret_cast<Expr*>(&aggregateInput);
    aggregateInputTarget.resno = 2;
    aggregateInputTarget.resjunk = false;

    auto aggregateArgTarget = TargetEntry{};
    aggregateArgTarget.xpr.type = T_TargetEntry;
    aggregateArgTarget.expr = reinterpret_cast<Expr*>(&aggregateInput);
    aggregateArgTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT4;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&aggregateArgTarget);
    aggregate.aggargtypes = list_make1_oid(INT4OID);
    aggregate.aggno = 1;

    auto aggregateTarget = TargetEntry{};
    aggregateTarget.xpr.type = T_TargetEntry;
    aggregateTarget.expr = reinterpret_cast<Expr*>(&aggregate);
    aggregateTarget.resno = 2;
    aggregateTarget.resjunk = false;

    auto leftScan = SeqScan{};
    leftScan.scan.plan.type = T_SeqScan;
    leftScan.scan.plan.targetlist = list_make2(&groupTarget, &aggregateInputTarget);

    auto rightScan = SeqScan{};
    rightScan.scan.plan.type = T_SeqScan;
    rightScan.scan.plan.targetlist = list_make2(&groupTarget, &aggregateInputTarget);

    auto join = NestLoop{};
    join.join.plan.type = T_NestLoop;
    join.join.plan.targetlist = list_make2(&groupTarget, &aggregateInputTarget);
    join.join.plan.lefttree = reinterpret_cast<Plan*>(&leftScan);
    join.join.plan.righttree = reinterpret_cast<Plan*>(&rightScan);

    auto sort = Sort{};
    AttrNumber sortColIdx[1]{1};
    Oid sortOperators[1]{Int4LessOperator};
    Oid sortCollations[1]{InvalidOid};
    bool nullsFirst[1]{false};
    sort.plan.type = T_Sort;
    sort.plan.targetlist = list_make2(&groupTarget, &aggregateInputTarget);
    sort.plan.lefttree = reinterpret_cast<Plan*>(&join);
    sort.numCols = 1;
    sort.sortColIdx = sortColIdx;
    sort.sortOperators = sortOperators;
    sort.collations = sortCollations;
    sort.nullsFirst = nullsFirst;

    auto agg = Agg{};
    AttrNumber grpColIdx[1]{1};
    Oid grpOperators[1]{Int4EqualOperator};
    Oid grpCollations[1]{InvalidOid};
    agg.plan.type = T_Agg;
    agg.plan.targetlist = list_make2(&groupTarget, &aggregateTarget);
    agg.plan.lefttree = reinterpret_cast<Plan*>(&sort);
    agg.aggstrategy = AGG_SORTED;
    agg.numCols = 1;
    agg.grpColIdx = grpColIdx;
    agg.grpOperators = grpOperators;
    agg.grpCollations = grpCollations;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&agg));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_having_only_aggregate) {
    AggPlanFixture fixture;

    auto argValue = makeTypedConst(INT4OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto havingAggregate = Aggref{};
    havingAggregate.xpr.type = T_Aggref;
    havingAggregate.aggfnoid = F_SUM_INT4;
    havingAggregate.aggtype = INT8OID;
    havingAggregate.aggcollid = InvalidOid;
    havingAggregate.inputcollid = InvalidOid;
    havingAggregate.args = list_make1(&argTarget);
    havingAggregate.aggargtypes = list_make1_oid(INT4OID);
    havingAggregate.aggno = 7;
    fixture.agg.plan.qual = list_make1(&havingAggregate);

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_having_only_aggregate_in_scalar_array_op) {
    AggPlanFixture fixture;

    auto argValue = makeTypedConst(INT4OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto havingAggregate = Aggref{};
    havingAggregate.xpr.type = T_Aggref;
    havingAggregate.aggfnoid = F_SUM_INT4;
    havingAggregate.aggtype = INT4OID;
    havingAggregate.aggcollid = InvalidOid;
    havingAggregate.inputcollid = InvalidOid;
    havingAggregate.args = list_make1(&argTarget);
    havingAggregate.aggargtypes = list_make1_oid(INT4OID);
    havingAggregate.aggno = 7;

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
    scalarArray.args = list_make2(&havingAggregate, &arrayExpr);
    fixture.agg.plan.qual = list_make1(&scalarArray);

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&fixture.agg));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
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

PGX_TEST_FN(query_analyzer_rejects_split_aggregate_plan) {
    AggPlanFixture fixture;
    fixture.agg.aggsplit = AGGSPLIT_INITIAL_SERIAL;

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

PGX_TEST_FN(query_analyzer_rejects_distinct_aggregate) {
    auto argValue = makeTypedConst(INT4OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;
    argTarget.ressortgroupref = 1;

    auto distinctClause = SortGroupClause{};
    distinctClause.tleSortGroupRef = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT4;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(INT4OID);
    aggregate.aggdistinct = list_make1(&distinctClause);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_split_aggregate_ref) {
    auto argValue = makeTypedConst(INT4OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT4;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(INT4OID);
    aggregate.aggsplit = AGGSPLIT_FINAL_DESERIAL;

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_aggregate_direct_args) {
    auto argValue = makeTypedConst(INT4OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto directValue = makeTypedConst(INT4OID);

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT4;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(INT4OID);
    aggregate.aggdirectargs = list_make1(&directValue);

    const auto result = pgx_lower::QueryAnalyzer::analyzeExprForTesting(reinterpret_cast<Node*>(&aggregate));
    REQUIRE(!result.isSupported());
    REQUIRE(result.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_expr_node);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_rejects_variadic_aggregate) {
    auto argValue = makeTypedConst(INT4OID);
    auto argTarget = TargetEntry{};
    argTarget.xpr.type = T_TargetEntry;
    argTarget.expr = reinterpret_cast<Expr*>(&argValue);
    argTarget.resno = 1;

    auto aggregate = Aggref{};
    aggregate.xpr.type = T_Aggref;
    aggregate.aggfnoid = F_SUM_INT4;
    aggregate.aggtype = INT8OID;
    aggregate.aggcollid = InvalidOid;
    aggregate.inputcollid = InvalidOid;
    aggregate.args = list_make1(&argTarget);
    aggregate.aggargtypes = list_make1_oid(INT4OID);
    aggregate.aggvariadic = true;

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

PGX_TEST_FN(query_analyzer_row_first_slice_surface) {
    auto idVar = makeTypedVar(INT8OID, 1);
    idVar.varno = 1;
    auto payloadVar = makeTypedVar(INT4OID, 2);
    payloadVar.varno = 1;
    auto filterVar = makeTypedVar(INT8OID, 1);
    filterVar.varno = 1;
    auto limitValue = makeTypedConst(INT8OID);
    auto filter = makeBinaryOperatorExpr("<", BOOLOID, reinterpret_cast<Node*>(&filterVar),
                                         reinterpret_cast<Node*>(&limitValue));

    auto idTarget = TargetEntry{};
    idTarget.xpr.type = T_TargetEntry;
    idTarget.expr = reinterpret_cast<Expr*>(&idVar);
    idTarget.resno = 1;
    idTarget.resjunk = false;
    auto payloadTarget = TargetEntry{};
    payloadTarget.xpr.type = T_TargetEntry;
    payloadTarget.expr = reinterpret_cast<Expr*>(&payloadVar);
    payloadTarget.resno = 2;
    payloadTarget.resjunk = false;

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = list_make2(&idTarget, &payloadTarget);
    scan.scan.plan.qual = list_make1(&filter);
    scan.scan.scanrelid = 1;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::row);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_first_slice_nullable_predicate_surface) {
    auto idTargetVar = makeTypedVar(INT8OID, 1);
    idTargetVar.varno = 1;
    auto payloadFilterVar = makeTypedVar(INT4OID, 2);
    payloadFilterVar.varno = 1;
    auto idFilterVar = makeTypedVar(INT8OID, 1);
    idFilterVar.varno = 1;
    auto idValue = makeTypedConst(INT8OID);
    auto idEquals = makeBinaryOperatorExpr("=", BOOLOID, reinterpret_cast<Node*>(&idFilterVar),
                                           reinterpret_cast<Node*>(&idValue));

    auto payloadIsNotNull = NullTest{};
    payloadIsNotNull.xpr.type = T_NullTest;
    payloadIsNotNull.arg = reinterpret_cast<Expr*>(&payloadFilterVar);
    payloadIsNotNull.nulltesttype = IS_NOT_NULL;
    payloadIsNotNull.argisrow = false;

    auto filter = BoolExpr{};
    filter.xpr.type = T_BoolExpr;
    filter.boolop = AND_EXPR;
    filter.args = list_make2(&payloadIsNotNull, &idEquals);

    auto idTarget = TargetEntry{};
    idTarget.xpr.type = T_TargetEntry;
    idTarget.expr = reinterpret_cast<Expr*>(&idTargetVar);
    idTarget.resno = 1;
    idTarget.resjunk = false;

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = list_make1(&idTarget);
    scan.scan.plan.qual = list_make1(&filter);
    scan.scan.scanrelid = 1;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::row);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_first_slice_rejects_not_equal_predicate) {
    auto idTargetVar = makeTypedVar(INT8OID, 1);
    idTargetVar.varno = 1;
    auto idFilterVar = makeTypedVar(INT8OID, 1);
    idFilterVar.varno = 1;
    auto idValue = makeTypedConst(INT8OID);
    auto notEquals = makeBinaryOperatorExpr("<>", BOOLOID, reinterpret_cast<Node*>(&idFilterVar),
                                            reinterpret_cast<Node*>(&idValue));

    auto idTarget = TargetEntry{};
    idTarget.xpr.type = T_TargetEntry;
    idTarget.expr = reinterpret_cast<Expr*>(&idTargetVar);
    idTarget.resno = 1;
    idTarget.resjunk = false;

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = list_make1(&idTarget);
    scan.scan.plan.qual = list_make1(&notEquals);
    scan.scan.scanrelid = 1;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::legacy);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_output_materialization_surface) {
    auto aTargetVar = makeTypedVar(INT4OID, 1);
    aTargetVar.varno = 1;
    auto aNullTestVar = makeTypedVar(INT4OID, 1);
    aNullTestVar.varno = 1;
    auto aCompareVar = makeTypedVar(INT4OID, 1);
    aCompareVar.varno = 1;
    auto cCompareVar = makeTypedVar(INT4OID, 3);
    cCompareVar.varno = 1;

    auto aIsNull = NullTest{};
    aIsNull.xpr.type = T_NullTest;
    aIsNull.arg = reinterpret_cast<Expr*>(&aNullTestVar);
    aIsNull.nulltesttype = IS_NULL;
    aIsNull.argisrow = false;

    auto aEqualsC = makeBinaryOperatorExpr("=", BOOLOID, reinterpret_cast<Node*>(&aCompareVar),
                                           reinterpret_cast<Node*>(&cCompareVar));

    auto aTarget = TargetEntry{};
    aTarget.xpr.type = T_TargetEntry;
    aTarget.expr = reinterpret_cast<Expr*>(&aTargetVar);
    aTarget.resno = 1;
    aTarget.resjunk = false;
    auto nullTarget = TargetEntry{};
    nullTarget.xpr.type = T_TargetEntry;
    nullTarget.expr = reinterpret_cast<Expr*>(&aIsNull);
    nullTarget.resno = 2;
    nullTarget.resjunk = false;
    auto compareTarget = TargetEntry{};
    compareTarget.xpr.type = T_TargetEntry;
    compareTarget.expr = reinterpret_cast<Expr*>(&aEqualsC);
    compareTarget.resno = 3;
    compareTarget.resjunk = false;

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = list_make3(&aTarget, &nullTarget, &compareTarget);
    scan.scan.scanrelid = 1;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::row);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_primitive_surface_matrix) {
    auto flag = makeTypedVar(BOOLOID, 1);
    flag.varno = 1;
    auto small = makeTypedVar(INT2OID, 2);
    small.varno = 1;
    auto i4 = makeTypedVar(INT4OID, 3);
    i4.varno = 1;
    auto i8 = makeTypedVar(INT8OID, 4);
    i8.varno = 1;
    auto f4 = makeTypedVar(FLOAT4OID, 5);
    f4.varno = 1;
    auto f8 = makeTypedVar(FLOAT8OID, 6);
    f8.varno = 1;
    auto amount = makeTypedVar(NUMERICOID, 7, Numeric12_2Typmod, InvalidOid);
    amount.varno = 1;
    auto d = makeTypedVar(DATEOID, 8);
    d.varno = 1;
    auto ts = makeTypedVar(TIMESTAMPOID, 9);
    ts.varno = 1;
    auto iv = makeTypedVar(INTERVALOID, 10);
    iv.varno = 1;
    auto text = makeTypedVar(TEXTOID, 11, -1, DEFAULT_COLLATION_OID);
    text.varno = 1;
    auto varchar = makeTypedVar(VARCHAROID, 12, Varchar16Typmod, DEFAULT_COLLATION_OID);
    varchar.varno = 1;
    auto bpchar = makeTypedVar(BPCHAROID, 13, Bpchar4Typmod, DEFAULT_COLLATION_OID);
    bpchar.varno = 1;
    auto filterVar = makeTypedVar(INT8OID, 4);
    filterVar.varno = 1;
    auto filterValue = makeTypedConst(INT8OID);
    auto filter = makeBinaryOperatorExpr("=", BOOLOID, reinterpret_cast<Node*>(&filterVar),
                                         reinterpret_cast<Node*>(&filterValue));

    auto flagTarget = makeOutputTarget(reinterpret_cast<Expr*>(&flag), 1);
    auto smallTarget = makeOutputTarget(reinterpret_cast<Expr*>(&small), 2);
    auto i4Target = makeOutputTarget(reinterpret_cast<Expr*>(&i4), 3);
    auto i8Target = makeOutputTarget(reinterpret_cast<Expr*>(&i8), 4);
    auto f4Target = makeOutputTarget(reinterpret_cast<Expr*>(&f4), 5);
    auto f8Target = makeOutputTarget(reinterpret_cast<Expr*>(&f8), 6);
    auto amountTarget = makeOutputTarget(reinterpret_cast<Expr*>(&amount), 7);
    auto dateTarget = makeOutputTarget(reinterpret_cast<Expr*>(&d), 8);
    auto timestampTarget = makeOutputTarget(reinterpret_cast<Expr*>(&ts), 9);
    auto intervalTarget = makeOutputTarget(reinterpret_cast<Expr*>(&iv), 10);
    auto textTarget = makeOutputTarget(reinterpret_cast<Expr*>(&text), 11);
    auto varcharTarget = makeOutputTarget(reinterpret_cast<Expr*>(&varchar), 12);
    auto bpcharTarget = makeOutputTarget(reinterpret_cast<Expr*>(&bpchar), 13);

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = NIL;
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &flagTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &smallTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &i4Target);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &i8Target);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &f4Target);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &f8Target);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &amountTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &dateTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &timestampTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &intervalTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &textTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &varcharTarget);
    scan.scan.plan.targetlist = lappend(scan.scan.plan.targetlist, &bpcharTarget);
    scan.scan.plan.qual = list_make1(&filter);
    scan.scan.scanrelid = 1;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::row);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_primitive_accepts_bare_bool_filter) {
    auto flagFilter = makeTypedVar(BOOLOID, 1);
    flagFilter.varno = 1;
    auto i8 = makeTypedVar(INT8OID, 2);
    i8.varno = 1;
    auto i8Target = makeOutputTarget(reinterpret_cast<Expr*>(&i8), 1);

    auto scan = SeqScan{};
    scan.scan.plan.type = T_SeqScan;
    scan.scan.plan.targetlist = list_make1(&i8Target);
    scan.scan.plan.qual = list_make1(&flagFilter);
    scan.scan.scanrelid = 1;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::row);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_primitive_unsupported_metadata) {
    auto bytea = makeTypedVar(BYTEAOID, 1);
    bytea.varno = 1;
    auto byteaTarget = makeOutputTarget(reinterpret_cast<Expr*>(&bytea), 1);
    auto byteaScan = SeqScan{};
    byteaScan.scan.plan.type = T_SeqScan;
    byteaScan.scan.plan.targetlist = list_make1(&byteaTarget);
    byteaScan.scan.scanrelid = 1;

    auto byteaResult = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&byteaScan));
    REQUIRE(!byteaResult.isSupported());
    REQUIRE(byteaResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);

    auto interval = makeTypedVar(INTERVALOID, 1, IntervalYearToMonthTypmod, InvalidOid);
    interval.varno = 1;
    auto intervalTarget = makeOutputTarget(reinterpret_cast<Expr*>(&interval), 1);
    auto intervalScan = SeqScan{};
    intervalScan.scan.plan.type = T_SeqScan;
    intervalScan.scan.plan.targetlist = list_make1(&intervalTarget);
    intervalScan.scan.scanrelid = 1;

    auto intervalResult = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&intervalScan));
    REQUIRE(!intervalResult.isSupported());
    REQUIRE(intervalResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_type);

    auto collatedText = makeTypedVar(TEXTOID, 1, -1, C_COLLATION_OID);
    collatedText.varno = 1;
    auto collatedTextTarget = makeOutputTarget(reinterpret_cast<Expr*>(&collatedText), 1);
    auto collatedScan = SeqScan{};
    collatedScan.scan.plan.type = T_SeqScan;
    collatedScan.scan.plan.targetlist = list_make1(&collatedTextTarget);
    collatedScan.scan.scanrelid = 1;

    auto collatedResult = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&collatedScan));
    REQUIRE(!collatedResult.isSupported());
    REQUIRE(collatedResult.primaryReason().kind == pgx_lower::UnsupportedReasonKind::unsupported_collation);
    REQUIRE(collatedResult.humanSummary().find("unsupported text collation") != std::string::npos);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_first_slice_legacy_for_supported_ineligible_plan) {
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

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(reinterpret_cast<Plan*>(&scan));
    REQUIRE(path == pgx_lower::LowerPath::legacy);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_row_first_slice_not_applicable_for_unsupported_plan) {
    auto plan = Plan{};
    plan.type = T_Invalid;

    const auto path = pgx_lower::QueryAnalyzer::classifyLowerPathForTesting(&plan);
    REQUIRE(path == pgx_lower::LowerPath::not_applicable);
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_limit_without_agg_metadata) {
    auto value = makeIntConst();
    auto limitCount = makeIntConst();
    auto target = TargetEntry{};
    target.xpr.type = T_TargetEntry;
    target.expr = reinterpret_cast<Expr*>(&value);
    target.resno = 1;
    target.resjunk = false;

    auto limit = Limit{};
    limit.plan.type = T_Limit;
    limit.plan.targetlist = list_make1(&target);
    limit.limitCount = reinterpret_cast<Node*>(&limitCount);
    limit.limitOption = LIMIT_OPTION_COUNT;

    const auto result = pgx_lower::QueryAnalyzer::analyzeNodeForTesting(reinterpret_cast<Plan*>(&limit));
    REQUIRE(result.isSupported());
    PG_RETURN_VOID();
}

PGX_TEST_FN(query_analyzer_accepts_index_scan_without_sort_metadata) {
    auto value = makeIntConst();
    auto target = TargetEntry{};
    target.xpr.type = T_TargetEntry;
    target.expr = reinterpret_cast<Expr*>(&value);
    target.resno = 1;
    target.resjunk = false;

    auto scan = IndexScan{};
    scan.scan.plan.type = T_IndexScan;
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
