#include "pgx-lower/frontend/SQL/query_analyzer.h"

#include "pgx_lower_constants.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "catalog/namespace.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_namespace_d.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_proc_d.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "nodes/primnodes.h"
#include "nodes/nodeFuncs.h"
#include "nodes/print.h"
#include "utils/lsyscache.h"

extern Oid g_jit_table_oid;
}
#include "pgx-lower/execution/postgres/executor_c.h"
#endif

#include <iterator>

#ifdef POSTGRESQL_EXTENSION

#endif

namespace pgx_lower {

auto unsupportedReasonKindName(const UnsupportedReasonKind kind) -> const char* {
    switch (kind) {
    case UnsupportedReasonKind::invalid: return "invalid";
    case UnsupportedReasonKind::unsupported_plan_node: return "unsupported_plan_node";
    case UnsupportedReasonKind::unsupported_expr_node: return "unsupported_expr_node";
    case UnsupportedReasonKind::unsupported_type: return "unsupported_type";
    case UnsupportedReasonKind::unsupported_operator: return "unsupported_operator";
    case UnsupportedReasonKind::unsupported_function: return "unsupported_function";
    case UnsupportedReasonKind::unsupported_collation: return "unsupported_collation";
    case UnsupportedReasonKind::missing_metadata: return "missing_metadata";
    }
    return "invalid";
}

AnalyzerResult::AnalyzerResult() {
    reasons_.push_back({UnsupportedReasonKind::invalid, "analyzer result was not explicitly constructed", {}});
}

auto AnalyzerResult::supported() -> AnalyzerResult {
    auto result = AnalyzerResult{};
    result.supported_ = true;
    result.reasons_.clear();
    return result;
}

auto AnalyzerResult::unsupported(UnsupportedReasonKind kind, std::string message, std::string location)
    -> AnalyzerResult {
    auto result = AnalyzerResult{};
    result.supported_ = false;
    result.reasons_.clear();
    result.reasons_.push_back({kind, std::move(message), std::move(location)});
    return result;
}

auto AnalyzerResult::isSupported() const -> bool {
    return supported_ && reasons_.empty();
}

auto AnalyzerResult::reasons() const -> const std::vector<UnsupportedReason>& {
    return reasons_;
}

auto AnalyzerResult::primaryReason() const -> const UnsupportedReason& {
    return reasons_.front();
}

auto AnalyzerResult::primaryReasonKindName() const -> std::string {
    return unsupportedReasonKindName(primaryReason().kind);
}

auto AnalyzerResult::humanSummary() const -> std::string {
    if (isSupported()) {
        return "supported";
    }

    const auto& reason = primaryReason();
    auto summary = std::string(unsupportedReasonKindName(reason.kind)) + ": " + reason.message;
    if (!reason.location.empty()) {
        summary += " at " + reason.location;
    }
    return summary;
}

auto AnalyzerResult::addUnsupportedReason(UnsupportedReasonKind kind, std::string message, std::string location) -> void {
    supported_ = false;
    reasons_.push_back({kind, std::move(message), std::move(location)});
}

#ifdef POSTGRESQL_EXTENSION

static auto mergeAnalyzerResult(AnalyzerResult& into, const AnalyzerResult& from) -> void {
    if (from.isSupported()) {
        return;
    }
    for (const auto& reason : from.reasons()) {
        into.addUnsupportedReason(reason.kind, reason.message, reason.location);
    }
}

static auto supportedOrUnsupported(const AnalyzerResult& result) -> AnalyzerResult {
    if (result.reasons().empty()) {
        return AnalyzerResult::supported();
    }
    return result;
}

static auto postgresTypeIsMLIRSupported(const Oid postgresType) -> bool {
    switch (postgresType) {
    case INT4OID:
    case INT8OID:
    case INT2OID:
    case FLOAT4OID:
    case FLOAT8OID:
    case BOOLOID:
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID:
    case NUMERICOID:
    case DATEOID:
    case TIMESTAMPOID:
    case INTERVALOID: return true;

    default: return false;
    }
}

static auto postgresTypeIsStringType(const Oid postgresType) -> bool {
    switch (postgresType) {
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: return true;
    default: return false;
    }
}

static auto postgresCollationIsSupported(const Oid collationOid) -> bool {
    return collationOid == InvalidOid || collationOid == DEFAULT_COLLATION_OID || collationOid == C_COLLATION_OID
           || collationOid == POSIX_COLLATION_OID;
}

struct PgFunctionSignature {
    const char* name;
    char kind;
    Oid resultType;
    int nargs;
    Oid argTypes[3];
};

static auto functionSignatureMatches(const Form_pg_proc proc, const PgFunctionSignature& signature) -> bool {
    if (proc->pronamespace != PG_CATALOG_NAMESPACE || proc->prokind != signature.kind
        || proc->prorettype != signature.resultType || proc->pronargs != signature.nargs
        || std::strcmp(NameStr(proc->proname), signature.name) != 0)
    {
        return false;
    }
    for (auto index = 0; index < signature.nargs; ++index) {
        if (proc->proargtypes.values[index] != signature.argTypes[index]) {
            return false;
        }
    }
    return true;
}

static auto catalogFunctionMatchesAny(const Oid functionOid, const PgFunctionSignature* signatures,
                                      const size_t signatureCount) -> bool {
    if (functionOid == InvalidOid) {
        return false;
    }
    const auto tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(functionOid));
    if (!HeapTupleIsValid(tuple)) {
        return false;
    }

    const auto proc = reinterpret_cast<Form_pg_proc>(GETSTRUCT(tuple));
    auto supported = false;
    for (size_t index = 0; index < signatureCount; ++index) {
        if (functionSignatureMatches(proc, signatures[index])) {
            supported = true;
            break;
        }
    }
    ReleaseSysCache(tuple);
    return supported;
}

static auto expressionListMatchesSignature(const List* expressions, const PgFunctionSignature& signature) -> bool {
    const auto expressionCount = expressions ? list_length(expressions) : 0;
    if (expressionCount != signature.nargs) {
        return false;
    }

    for (auto index = 0; index < signature.nargs; ++index) {
        const auto* expr = static_cast<const Node*>(lfirst(list_nth_cell(expressions, index)));
        if (!expr || exprType(const_cast<Node*>(expr)) != signature.argTypes[index]) {
            return false;
        }
    }
    return true;
}

static auto functionExprMatchesAny(const FuncExpr* func, const PgFunctionSignature* signatures,
                                   const size_t signatureCount) -> bool {
    if (!func || func->funcid == InvalidOid) {
        return false;
    }
    const auto tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func->funcid));
    if (!HeapTupleIsValid(tuple)) {
        return false;
    }

    const auto proc = reinterpret_cast<Form_pg_proc>(GETSTRUCT(tuple));
    auto supported = false;
    for (size_t index = 0; index < signatureCount; ++index) {
        const auto& signature = signatures[index];
        if (functionSignatureMatches(proc, signature) && func->funcresulttype == signature.resultType
            && expressionListMatchesSignature(func->args, signature))
        {
            supported = true;
            break;
        }
    }
    ReleaseSysCache(tuple);
    return supported;
}

static auto operatorExprMatchesCatalog(const OpExpr* op) -> bool {
    if (!op || op->opno == InvalidOid || op->opresulttype == InvalidOid || !op->args || list_length(op->args) != 2) {
        return false;
    }

    const auto tuple = SearchSysCache1(OPEROID, ObjectIdGetDatum(op->opno));
    if (!HeapTupleIsValid(tuple)) {
        return false;
    }

    const auto oper = reinterpret_cast<Form_pg_operator>(GETSTRUCT(tuple));
    auto matches = oper->oprnamespace == PG_CATALOG_NAMESPACE && oper->oprkind == 'b'
                   && oper->oprresult == op->opresulttype;
    if (matches) {
        const auto* lhs = static_cast<const Node*>(lfirst(list_nth_cell(op->args, 0)));
        const auto* rhs = static_cast<const Node*>(lfirst(list_nth_cell(op->args, 1)));
        matches = lhs != nullptr && rhs != nullptr && exprType(const_cast<Node*>(lhs)) == oper->oprleft
                  && exprType(const_cast<Node*>(rhs)) == oper->oprright;
    }
    ReleaseSysCache(tuple);
    return matches;
}

struct PgOperatorTypeSignature {
    Oid resultType;
    Oid leftType;
    Oid rightType;
};

static auto operatorNameMatchesAny(const char* operatorName, const char* const* names, const size_t nameCount) -> bool {
    if (!operatorName) {
        return false;
    }
    for (size_t index = 0; index < nameCount; ++index) {
        if (std::strcmp(operatorName, names[index]) == 0) {
            return true;
        }
    }
    return false;
}

static auto operatorTypeSignatureMatchesAny(const Oid resultType, const Oid leftType, const Oid rightType,
                                            const PgOperatorTypeSignature* signatures, const size_t signatureCount)
    -> bool {
    for (size_t index = 0; index < signatureCount; ++index) {
        const auto& signature = signatures[index];
        if (signature.resultType == resultType && signature.leftType == leftType && signature.rightType == rightType) {
            return true;
        }
    }
    return false;
}

static auto operatorCatalogMatches(const Oid operatorOid, const Oid resultType, const Oid leftType, const Oid rightType)
    -> bool {
    if (operatorOid == InvalidOid || resultType == InvalidOid || leftType == InvalidOid || rightType == InvalidOid) {
        return false;
    }

    const auto tuple = SearchSysCache1(OPEROID, ObjectIdGetDatum(operatorOid));
    if (!HeapTupleIsValid(tuple)) {
        return false;
    }

    const auto oper = reinterpret_cast<Form_pg_operator>(GETSTRUCT(tuple));
    const auto matches = oper->oprnamespace == PG_CATALOG_NAMESPACE && oper->oprkind == 'b'
                         && oper->oprresult == resultType && oper->oprleft == leftType && oper->oprright == rightType;
    ReleaseSysCache(tuple);
    return matches;
}

static constexpr const char* equalityOperatorNames[] = {"=", "<>"};
static constexpr const char* groupingEqualityOperatorNames[] = {"="};
static constexpr const char* orderingOperatorNames[] = {"<", "<=", ">", ">="};
static constexpr const char* strictOrderingOperatorNames[] = {"<", ">"};
static constexpr const char* arithmeticOperatorNames[] = {"+", "-", "*", "/"};
static constexpr const char* likeOperatorNames[] = {"~~", "!~~"};

static constexpr PgOperatorTypeSignature supportedEqualityOperatorSignatures[] = {
    {BOOLOID, BOOLOID, BOOLOID},         {BOOLOID, INT2OID, INT2OID},      {BOOLOID, INT4OID, INT4OID},
    {BOOLOID, INT8OID, INT8OID},         {BOOLOID, INT2OID, INT4OID},      {BOOLOID, INT4OID, INT2OID},
    {BOOLOID, INT2OID, INT8OID},         {BOOLOID, INT8OID, INT2OID},      {BOOLOID, INT4OID, INT8OID},
    {BOOLOID, INT8OID, INT4OID},         {BOOLOID, FLOAT4OID, FLOAT4OID},  {BOOLOID, FLOAT8OID, FLOAT8OID},
    {BOOLOID, FLOAT4OID, FLOAT8OID},     {BOOLOID, FLOAT8OID, FLOAT4OID},  {BOOLOID, NUMERICOID, NUMERICOID},
    {BOOLOID, TEXTOID, TEXTOID},         {BOOLOID, BPCHAROID, BPCHAROID},  {BOOLOID, DATEOID, DATEOID},
    {BOOLOID, DATEOID, TIMESTAMPOID},    {BOOLOID, TIMESTAMPOID, DATEOID}, {BOOLOID, TIMESTAMPOID, TIMESTAMPOID},
    {BOOLOID, INTERVALOID, INTERVALOID},
};

static constexpr PgOperatorTypeSignature supportedOrderingOperatorSignatures[] = {
    {BOOLOID, INT2OID, INT2OID},      {BOOLOID, INT4OID, INT4OID},           {BOOLOID, INT8OID, INT8OID},
    {BOOLOID, INT2OID, INT4OID},      {BOOLOID, INT4OID, INT2OID},           {BOOLOID, INT2OID, INT8OID},
    {BOOLOID, INT8OID, INT2OID},      {BOOLOID, INT4OID, INT8OID},           {BOOLOID, INT8OID, INT4OID},
    {BOOLOID, FLOAT4OID, FLOAT4OID},  {BOOLOID, FLOAT8OID, FLOAT8OID},       {BOOLOID, FLOAT4OID, FLOAT8OID},
    {BOOLOID, FLOAT8OID, FLOAT4OID},  {BOOLOID, NUMERICOID, NUMERICOID},     {BOOLOID, TEXTOID, TEXTOID},
    {BOOLOID, BPCHAROID, BPCHAROID},  {BOOLOID, DATEOID, DATEOID},           {BOOLOID, DATEOID, TIMESTAMPOID},
    {BOOLOID, TIMESTAMPOID, DATEOID}, {BOOLOID, TIMESTAMPOID, TIMESTAMPOID}, {BOOLOID, INTERVALOID, INTERVALOID},
};

static constexpr PgOperatorTypeSignature supportedArithmeticOperatorSignatures[] = {
    {INT2OID, INT2OID, INT2OID},
    {INT4OID, INT4OID, INT4OID},
    {INT8OID, INT8OID, INT8OID},
    {INT4OID, INT2OID, INT4OID},
    {INT4OID, INT4OID, INT2OID},
    {INT8OID, INT2OID, INT8OID},
    {INT8OID, INT8OID, INT2OID},
    {INT8OID, INT4OID, INT8OID},
    {INT8OID, INT8OID, INT4OID},
    {FLOAT4OID, FLOAT4OID, FLOAT4OID},
    {FLOAT8OID, FLOAT8OID, FLOAT8OID},
    {FLOAT8OID, FLOAT4OID, FLOAT8OID},
    {FLOAT8OID, FLOAT8OID, FLOAT4OID},
    {NUMERICOID, NUMERICOID, NUMERICOID},
    {TIMESTAMPOID, DATEOID, INTERVALOID},
    {TIMESTAMPOID, INTERVALOID, DATEOID},
    {TIMESTAMPOID, TIMESTAMPOID, INTERVALOID},
    {TIMESTAMPOID, INTERVALOID, TIMESTAMPOID},
    {INTERVALOID, TIMESTAMPOID, TIMESTAMPOID},
    {DATEOID, DATEOID, INT4OID},
    {DATEOID, INT4OID, DATEOID},
    {INTERVALOID, INTERVALOID, INTERVALOID},
};

static constexpr PgOperatorTypeSignature supportedLikeOperatorSignatures[] = {
    {BOOLOID, TEXTOID, TEXTOID},
    {BOOLOID, BPCHAROID, TEXTOID},
};

static auto
operatorSignatureIsLowerable(const Oid operatorOid, const Oid resultType, const Oid lhsType, const Oid rhsType) -> bool {
    const char* name = get_opname(operatorOid);
    if (!name) {
        return false;
    }
    const auto supportsEquality = operatorNameMatchesAny(name, equalityOperatorNames, std::size(equalityOperatorNames));
    const auto supportsOrdering = operatorNameMatchesAny(name, orderingOperatorNames, std::size(orderingOperatorNames));
    const auto supportsArithmetic = operatorNameMatchesAny(name, arithmeticOperatorNames,
                                                           std::size(arithmeticOperatorNames));
    const auto supportsLike = operatorNameMatchesAny(name, likeOperatorNames, std::size(likeOperatorNames));
    pfree(const_cast<char*>(name));

    if (supportsEquality
        && operatorTypeSignatureMatchesAny(resultType, lhsType, rhsType, supportedEqualityOperatorSignatures,
                                           std::size(supportedEqualityOperatorSignatures)))
    {
        return true;
    }
    if (supportsOrdering
        && operatorTypeSignatureMatchesAny(resultType, lhsType, rhsType, supportedOrderingOperatorSignatures,
                                           std::size(supportedOrderingOperatorSignatures)))
    {
        return true;
    }
    if (supportsArithmetic
        && operatorTypeSignatureMatchesAny(resultType, lhsType, rhsType, supportedArithmeticOperatorSignatures,
                                           std::size(supportedArithmeticOperatorSignatures)))
    {
        return true;
    }
    return supportsLike
           && operatorTypeSignatureMatchesAny(resultType, lhsType, rhsType, supportedLikeOperatorSignatures,
                                              std::size(supportedLikeOperatorSignatures));
}

static auto operatorSignatureIsLowerable(const OpExpr* op) -> bool {
    const auto* lhs = static_cast<const Node*>(lfirst(list_nth_cell(op->args, 0)));
    const auto* rhs = static_cast<const Node*>(lfirst(list_nth_cell(op->args, 1)));
    return operatorSignatureIsLowerable(op->opno, op->opresulttype, exprType(const_cast<Node*>(lhs)),
                                        exprType(const_cast<Node*>(rhs)));
}

static auto sortOperatorMatchesTargetType(const Oid operatorOid, const Oid keyType) -> bool {
    if (operatorOid == InvalidOid || keyType == InvalidOid) {
        return false;
    }

    const auto tuple = SearchSysCache1(OPEROID, ObjectIdGetDatum(operatorOid));
    if (!HeapTupleIsValid(tuple)) {
        return false;
    }

    const auto oper = reinterpret_cast<Form_pg_operator>(GETSTRUCT(tuple));
    const auto matches = oper->oprnamespace == PG_CATALOG_NAMESPACE && oper->oprkind == 'b'
                         && oper->oprresult == BOOLOID && oper->oprleft == keyType && oper->oprright == keyType
                         && operatorNameMatchesAny(NameStr(oper->oprname), strictOrderingOperatorNames,
                                                   std::size(strictOrderingOperatorNames))
                         && operatorTypeSignatureMatchesAny(oper->oprresult, oper->oprleft, oper->oprright,
                                                            supportedOrderingOperatorSignatures,
                                                            std::size(supportedOrderingOperatorSignatures));
    ReleaseSysCache(tuple);
    return matches;
}

static auto sortOperatorDirection(const Oid operatorOid, bool& descending) -> bool {
    const char* name = get_opname(operatorOid);
    if (!name) {
        return false;
    }

    const auto operatorName = std::string(name);
    pfree(const_cast<char*>(name));
    if (operatorName == "<") {
        descending = false;
        return true;
    }
    if (operatorName == ">") {
        descending = true;
        return true;
    }
    return false;
}

static auto scalarArrayOperatorSignatureIsLowerable(const Oid operatorOid, const Oid lhsType, const Oid rhsType) -> bool {
    const char* name = get_opname(operatorOid);
    if (!name) {
        return false;
    }
    const auto supportsEquality = operatorNameMatchesAny(name, equalityOperatorNames, std::size(equalityOperatorNames));
    pfree(const_cast<char*>(name));
    return supportsEquality
           && operatorTypeSignatureMatchesAny(BOOLOID, lhsType, rhsType, supportedEqualityOperatorSignatures,
                                              std::size(supportedEqualityOperatorSignatures));
}

static auto scalarArrayElementType(const Node* rightNode) -> Oid {
    if (!rightNode) {
        return InvalidOid;
    }
    if (nodeTag(rightNode) == T_ArrayExpr) {
        const auto* arrayExpr = reinterpret_cast<const ArrayExpr*>(rightNode);
        return arrayExpr->element_typeid;
    }
    if (nodeTag(rightNode) == T_Const) {
        const auto* constExpr = reinterpret_cast<const Const*>(rightNode);
        switch (constExpr->consttype) {
        case INT4ARRAYOID: return INT4OID;
        default: return InvalidOid;
        }
    }
    return InvalidOid;
}

static constexpr PgFunctionSignature supportedScalarFunctions[] = {
    {"upper", PROKIND_FUNCTION, TEXTOID, 1, {TEXTOID, InvalidOid, InvalidOid}},
    {"lower", PROKIND_FUNCTION, TEXTOID, 1, {TEXTOID, InvalidOid, InvalidOid}},
    {"substr", PROKIND_FUNCTION, TEXTOID, 2, {TEXTOID, INT4OID, InvalidOid}},
    {"substr", PROKIND_FUNCTION, TEXTOID, 3, {TEXTOID, INT4OID, INT4OID}},
    {"substring", PROKIND_FUNCTION, TEXTOID, 2, {TEXTOID, INT4OID, InvalidOid}},
    {"substring", PROKIND_FUNCTION, TEXTOID, 3, {TEXTOID, INT4OID, INT4OID}},
    {"numeric", PROKIND_FUNCTION, NUMERICOID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"numeric", PROKIND_FUNCTION, NUMERICOID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"numeric", PROKIND_FUNCTION, NUMERICOID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"numeric", PROKIND_FUNCTION, NUMERICOID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"numeric", PROKIND_FUNCTION, NUMERICOID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"numeric", PROKIND_FUNCTION, NUMERICOID, 2, {NUMERICOID, INT4OID, InvalidOid}},
    {"varchar", PROKIND_FUNCTION, VARCHAROID, 3, {VARCHAROID, INT4OID, BOOLOID}},
    {"text", PROKIND_FUNCTION, TEXTOID, 1, {BPCHAROID, InvalidOid, InvalidOid}},
    {"int4", PROKIND_FUNCTION, INT4OID, 1, {BOOLOID, InvalidOid, InvalidOid}},
    {"int4", PROKIND_FUNCTION, INT4OID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"int4", PROKIND_FUNCTION, INT4OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"int4", PROKIND_FUNCTION, INT4OID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"int4", PROKIND_FUNCTION, INT4OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"int4", PROKIND_FUNCTION, INT4OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"int8", PROKIND_FUNCTION, INT8OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"int8", PROKIND_FUNCTION, INT8OID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"int8", PROKIND_FUNCTION, INT8OID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"int8", PROKIND_FUNCTION, INT8OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"int8", PROKIND_FUNCTION, INT8OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"float4", PROKIND_FUNCTION, FLOAT4OID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"float4", PROKIND_FUNCTION, FLOAT4OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"float4", PROKIND_FUNCTION, FLOAT4OID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"float4", PROKIND_FUNCTION, FLOAT4OID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"float4", PROKIND_FUNCTION, FLOAT4OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"float8", PROKIND_FUNCTION, FLOAT8OID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"float8", PROKIND_FUNCTION, FLOAT8OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"float8", PROKIND_FUNCTION, FLOAT8OID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"float8", PROKIND_FUNCTION, FLOAT8OID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"float8", PROKIND_FUNCTION, FLOAT8OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
};

static constexpr PgFunctionSignature supportedAggregates[] = {
    {"count", PROKIND_AGGREGATE, INT8OID, 0, {InvalidOid, InvalidOid, InvalidOid}},
    {"count", PROKIND_AGGREGATE, INT8OID, 1, {ANYOID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, NUMERICOID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, INT8OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, INT8OID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, FLOAT4OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, FLOAT8OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, INTERVALOID, 1, {INTERVALOID, InvalidOid, InvalidOid}},
    {"sum", PROKIND_AGGREGATE, NUMERICOID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, NUMERICOID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, NUMERICOID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, NUMERICOID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, FLOAT8OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, FLOAT8OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, INTERVALOID, 1, {INTERVALOID, InvalidOid, InvalidOid}},
    {"avg", PROKIND_AGGREGATE, NUMERICOID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, INT8OID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, INT2OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, INT4OID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, FLOAT4OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, FLOAT8OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, NUMERICOID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, TEXTOID, 1, {TEXTOID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, BPCHAROID, 1, {BPCHAROID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, DATEOID, 1, {DATEOID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, TIMESTAMPOID, 1, {TIMESTAMPOID, InvalidOid, InvalidOid}},
    {"min", PROKIND_AGGREGATE, INTERVALOID, 1, {INTERVALOID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, INT8OID, 1, {INT8OID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, INT2OID, 1, {INT2OID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, INT4OID, 1, {INT4OID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, FLOAT4OID, 1, {FLOAT4OID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, FLOAT8OID, 1, {FLOAT8OID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, NUMERICOID, 1, {NUMERICOID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, TEXTOID, 1, {TEXTOID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, BPCHAROID, 1, {BPCHAROID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, DATEOID, 1, {DATEOID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, TIMESTAMPOID, 1, {TIMESTAMPOID, InvalidOid, InvalidOid}},
    {"max", PROKIND_AGGREGATE, INTERVALOID, 1, {INTERVALOID, InvalidOid, InvalidOid}},
};

static auto postgresFunctionName(const Oid functionOid) -> std::string {
    if (functionOid == InvalidOid) {
        return {};
    }
    const char* name = get_func_name(functionOid);
    if (!name) {
        return {};
    }
    auto functionName = std::string(name);
    pfree(const_cast<char*>(name));
    return functionName;
}

static auto targetListEntryType(const List* targetList, const AttrNumber column) -> Oid {
    if (!targetList) {
        return InvalidOid;
    }

    ListCell* lc = nullptr;
    foreach (lc, targetList) {
        const auto* tle = static_cast<const TargetEntry*>(lfirst(lc));
        if (!tle || tle->resjunk || tle->resno != column || !tle->expr) {
            continue;
        }
        return exprType(const_cast<Node*>(reinterpret_cast<const Node*>(tle->expr)));
    }
    return InvalidOid;
}

static auto groupingOperatorMatchesTargetType(const Oid operatorOid, const Oid keyType) -> bool {
    if (operatorOid == InvalidOid || keyType == InvalidOid) {
        return false;
    }

    const auto tuple = SearchSysCache1(OPEROID, ObjectIdGetDatum(operatorOid));
    if (!HeapTupleIsValid(tuple)) {
        return false;
    }

    const auto oper = reinterpret_cast<Form_pg_operator>(GETSTRUCT(tuple));
    const auto matches = oper->oprnamespace == PG_CATALOG_NAMESPACE && oper->oprkind == 'b'
                         && oper->oprresult == BOOLOID && oper->oprleft == keyType && oper->oprright == keyType
                         && operatorNameMatchesAny(NameStr(oper->oprname), groupingEqualityOperatorNames,
                                                   std::size(groupingEqualityOperatorNames))
                         && operatorTypeSignatureMatchesAny(oper->oprresult, oper->oprleft, oper->oprright,
                                                            supportedEqualityOperatorSignatures,
                                                            std::size(supportedEqualityOperatorSignatures));
    ReleaseSysCache(tuple);
    return matches;
}

static auto analyzeSortMetadata(const Sort* sort, const std::string& location) -> AnalyzerResult {
    auto result = AnalyzerResult::supported();
    if (!sort) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "sort node is null", location);
    }
    if (sort->numCols < 0
        || (sort->numCols > 0 && (!sort->sortColIdx || !sort->sortOperators || !sort->collations || !sort->nullsFirst)))
    {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "sort metadata is incomplete",
                                           location);
    }

    for (auto index = 0; index < sort->numCols; ++index) {
        const auto itemLocation = location + ".sortCol[" + std::to_string(index) + "]";
        if (sort->collations && !postgresCollationIsSupported(sort->collations[index])) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation, "unsupported sort collation",
                                        itemLocation);
        }

        const auto keyType = targetListEntryType(sort->plan.targetlist, sort->sortColIdx[index]);
        if (keyType == InvalidOid) {
            result.addUnsupportedReason(UnsupportedReasonKind::missing_metadata, "sort key target metadata is missing",
                                        itemLocation);
            continue;
        }
        if (!sortOperatorMatchesTargetType(sort->sortOperators[index], keyType)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_operator,
                                        "unsupported sort operator OID " + std::to_string(sort->sortOperators[index]),
                                        itemLocation);
            continue;
        }

        auto descending = false;
        if (!sortOperatorDirection(sort->sortOperators[index], descending)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_operator,
                                        "unsupported sort operator OID " + std::to_string(sort->sortOperators[index]),
                                        itemLocation);
            continue;
        }
        if (sort->nullsFirst[index] != descending) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_plan_node,
                                        "unsupported explicit sort null ordering", itemLocation);
        }
    }
    return supportedOrUnsupported(result);
}

static auto analyzeAggMetadata(const Agg* agg, const std::string& location) -> AnalyzerResult {
    auto result = AnalyzerResult::supported();
    if (!agg) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "aggregate node is null", location);
    }
    if (agg->numCols < 0
        || (agg->numCols > 0 && (!agg->grpColIdx || !agg->grpOperators || !agg->grpCollations)))
    {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata,
                                           "aggregate grouping metadata is incomplete", location);
    }

    for (auto index = 0; index < agg->numCols; ++index) {
        const auto itemLocation = location + ".groupCol[" + std::to_string(index) + "]";
        if (agg->grpCollations && !postgresCollationIsSupported(agg->grpCollations[index])) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation,
                                        "unsupported aggregate grouping collation", itemLocation);
        }

        const auto keyType = targetListEntryType(agg->plan.targetlist, agg->grpColIdx[index]);
        if (keyType == InvalidOid) {
            result.addUnsupportedReason(UnsupportedReasonKind::missing_metadata,
                                        "aggregate grouping key target metadata is missing", itemLocation);
            continue;
        }
        if (!groupingOperatorMatchesTargetType(agg->grpOperators[index], keyType)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_operator,
                                        "unsupported aggregate grouping operator OID "
                                            + std::to_string(agg->grpOperators[index]),
                                        itemLocation);
        }
    }
    return supportedOrUnsupported(result);
}

auto QueryAnalyzer::analyzePlan(const PlannedStmt* stmt) -> AnalyzerResult {
    if (!stmt) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::invalid, "planned statement is null", "PlannedStmt");
    }
    if (!stmt->planTree) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::invalid, "plan tree is null", "PlannedStmt.planTree");
    }
    if (!checkCommandType(stmt)) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::unsupported_plan_node,
                                           "only SELECT statements are supported", "PlannedStmt.commandType");
    }

    auto result = analyzeNode(stmt->planTree, "Plan");
    mergeAnalyzerResult(result, analyzePlanTargetTypes(stmt->planTree, "Plan.targetlist"));
    return supportedOrUnsupported(result);
}

auto QueryAnalyzer::analyzeNode(const Plan* plan, std::string location) -> AnalyzerResult {
    if (!plan) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "plan node is null",
                                           std::move(location));
    }

    auto result = AnalyzerResult::supported();

    switch (nodeTag(plan)) {
    case T_SeqScan:
    case T_NestLoop:
    case T_MergeJoin:
    case T_HashJoin: break;
    case T_Sort: mergeAnalyzerResult(result, analyzeSortMetadata(reinterpret_cast<const Sort*>(plan), location)); break;
    case T_Limit:
    case T_Agg:
        mergeAnalyzerResult(result, analyzeAggMetadata(reinterpret_cast<const Agg*>(plan), location));
        break;
    case T_Material:
    case T_Hash: break;
    case T_ProjectSet:
        mergeAnalyzerResult(result, analyzeExprList(plan->qual, location + ".qual"));
        mergeAnalyzerResult(result, analyzeTargetList(plan->targetlist, location + ".targetlist"));
        if (result.isSupported()) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_plan_node,
                                        "unsupported plan node tag " + std::to_string(nodeTag(plan)), location);
        }
        return supportedOrUnsupported(result);
    case T_SubqueryScan: {
        const auto* subqueryScan = reinterpret_cast<const SubqueryScan*>(plan);
        mergeAnalyzerResult(result, analyzeNode(subqueryScan->subplan, location + ".subplan"));
        break;
    }

    default:
        result.addUnsupportedReason(UnsupportedReasonKind::unsupported_plan_node,
                                    "unsupported plan node tag " + std::to_string(nodeTag(plan)), location);
        return result;
    }

    mergeAnalyzerResult(result, analyzeExprList(plan->qual, location + ".qual"));
    mergeAnalyzerResult(result, analyzeTargetList(plan->targetlist, location + ".targetlist"));

    if (plan->lefttree) {
        mergeAnalyzerResult(result, analyzeNode(plan->lefttree, location + ".lefttree"));
    }
    if (plan->righttree) {
        mergeAnalyzerResult(result, analyzeNode(plan->righttree, location + ".righttree"));
    }

    return supportedOrUnsupported(result);
}

auto QueryAnalyzer::analyzeTargetList(const List* targetList, const std::string& location) -> AnalyzerResult {
    auto result = AnalyzerResult::supported();
    if (!targetList) {
        return result;
    }

    ListCell* lc = nullptr;
    auto index = 0;
    foreach (lc, targetList) {
        const auto* tle = static_cast<const TargetEntry*>(lfirst(lc));
        const auto exprLocation = location + "[" + std::to_string(index) + "]";
        if (!tle) {
            result.addUnsupportedReason(UnsupportedReasonKind::missing_metadata, "target entry is null", exprLocation);
            ++index;
            continue;
        }
        if (tle->resjunk) {
            ++index;
            continue;
        }
        if (!tle->expr) {
            result.addUnsupportedReason(UnsupportedReasonKind::missing_metadata, "target expression is null",
                                        exprLocation + ".expr");
            ++index;
            continue;
        }
        mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(tle->expr), exprLocation + ".expr"));
        mergeAnalyzerResult(result, analyzeExprType(reinterpret_cast<const Node*>(tle->expr), exprLocation + ".type"));
        ++index;
    }
    return supportedOrUnsupported(result);
}

auto QueryAnalyzer::analyzeExprList(const List* expressions, const std::string& location) -> AnalyzerResult {
    auto result = AnalyzerResult::supported();
    if (!expressions) {
        return result;
    }

    ListCell* lc = nullptr;
    auto index = 0;
    foreach (lc, expressions) {
        const auto* expr = static_cast<const Node*>(lfirst(lc));
        mergeAnalyzerResult(result, analyzeExpr(expr, location + "[" + std::to_string(index) + "]"));
        ++index;
    }
    return supportedOrUnsupported(result);
}

auto QueryAnalyzer::analyzePlanTargetTypes(const Plan* plan, std::string location) -> AnalyzerResult {
    if (!plan) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "plan is null", std::move(location));
    }
    if (!plan->targetlist) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "plan targetlist is null",
                                           std::move(location));
    }
    return analyzeTargetList(plan->targetlist, location);
}

auto QueryAnalyzer::analyzeExprType(const Node* expr, std::string location) -> AnalyzerResult {
    if (!expr) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "expression is null",
                                           std::move(location));
    }

    const auto typeOid = exprType(const_cast<Node*>(expr));
    if (typeOid == InvalidOid) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::missing_metadata, "expression type OID is invalid",
                                           std::move(location));
    }
    if (!isTypeSupportedByMLIR(typeOid)) {
        return AnalyzerResult::unsupported(UnsupportedReasonKind::unsupported_type,
                                           "unsupported PostgreSQL type OID " + std::to_string(typeOid),
                                           std::move(location));
    }
    return AnalyzerResult::supported();
}

auto QueryAnalyzer::analyzeExpr(const Node* expr, const std::string& location) -> AnalyzerResult {
    if (!expr) {
        return AnalyzerResult::supported();
    }

    auto result = AnalyzerResult::supported();

    switch (nodeTag(expr)) {
    case T_Var:
    case T_Const:
    case T_Param:
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);

    case T_FuncExpr: {
        const auto* func = reinterpret_cast<const FuncExpr*>(expr);
        if (!isFunctionSupported(func)) {
            const auto functionName = postgresFunctionName(func->funcid);
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_function,
                                        functionName.empty() ? "unsupported function OID " + std::to_string(func->funcid)
                                                             : "unsupported function " + functionName + "()",
                                        location);
        }
        if (!isCollationSupported(func->inputcollid) || !isCollationSupported(func->funccollid)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation, "unsupported function collation",
                                        location);
        }
        mergeAnalyzerResult(result, analyzeExprList(func->args, location + ".args"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_OpExpr: {
        const auto* op = reinterpret_cast<const OpExpr*>(expr);
        if (!isOperatorSupported(op)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_operator,
                                        "unsupported operator OID " + std::to_string(op->opno), location);
        }
        if (!isCollationSupported(op->inputcollid) || !isCollationSupported(op->opcollid)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation, "unsupported operator collation",
                                        location);
        }
        mergeAnalyzerResult(result, analyzeExprList(op->args, location + ".args"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_BoolExpr: {
        const auto* boolExpr = reinterpret_cast<const BoolExpr*>(expr);
        mergeAnalyzerResult(result, analyzeExprList(boolExpr->args, location + ".args"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_CoalesceExpr: {
        const auto* coalesce = reinterpret_cast<const CoalesceExpr*>(expr);
        if (!isCollationSupported(coalesce->coalescecollid)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation, "unsupported COALESCE collation",
                                        location);
        }
        mergeAnalyzerResult(result, analyzeExprList(coalesce->args, location + ".args"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_ScalarArrayOpExpr: {
        const auto* scalarArray = reinterpret_cast<const ScalarArrayOpExpr*>(expr);
        if (!isCollationSupported(scalarArray->inputcollid)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation,
                                        "unsupported ScalarArrayOpExpr collation", location);
        }
        if (!scalarArray->args || list_length(scalarArray->args) != 2) {
            result.addUnsupportedReason(UnsupportedReasonKind::missing_metadata,
                                        "ScalarArrayOpExpr requires two arguments", location);
            return supportedOrUnsupported(result);
        }

        const auto* leftNode = static_cast<const Node*>(lfirst(list_nth_cell(scalarArray->args, 0)));
        const auto* rightNode = static_cast<const Node*>(lfirst(list_nth_cell(scalarArray->args, 1)));
        mergeAnalyzerResult(result, analyzeExpr(leftNode, location + ".left"));

        const auto elementType = scalarArrayElementType(rightNode);
        if (elementType == InvalidOid) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_expr_node,
                                        "unsupported ScalarArrayOpExpr array operand", location + ".right");
        } else if (nodeTag(rightNode) == T_ArrayExpr) {
            const auto* arrayExpr = reinterpret_cast<const ArrayExpr*>(rightNode);
            mergeAnalyzerResult(result, analyzeExprList(arrayExpr->elements, location + ".right.elements"));
        }

        const auto leftType = leftNode ? exprType(const_cast<Node*>(leftNode)) : InvalidOid;
        if (!operatorCatalogMatches(scalarArray->opno, BOOLOID, leftType, elementType)
            || !scalarArrayOperatorSignatureIsLowerable(scalarArray->opno, leftType, elementType))
        {
            result.addUnsupportedReason(
                UnsupportedReasonKind::unsupported_operator,
                "unsupported ScalarArrayOpExpr operator OID " + std::to_string(scalarArray->opno), location);
        }
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_RelabelType: {
        const auto* relabel = reinterpret_cast<const RelabelType*>(expr);
        if (!isCollationSupported(relabel->resultcollid)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation, "unsupported relabel collation",
                                        location);
        }
        mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(relabel->arg), location + ".arg"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_CoerceViaIO: {
        const auto* coerce = reinterpret_cast<const CoerceViaIO*>(expr);
        if (!isCollationSupported(coerce->resultcollid)) {
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_collation,
                                        "unsupported CoerceViaIO collation", location);
        }
        if (coerce->arg) {
            const auto inputType = exprType(reinterpret_cast<Node*>(coerce->arg));
            if (!postgresTypeIsStringType(inputType) || !postgresTypeIsStringType(coerce->resulttype)) {
                result.addUnsupportedReason(UnsupportedReasonKind::unsupported_expr_node,
                                            "unsupported CoerceViaIO from type OID " + std::to_string(inputType)
                                                + " to type OID " + std::to_string(coerce->resulttype),
                                            location);
            }
            mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(coerce->arg), location + ".arg"));
        } else {
            result.addUnsupportedReason(UnsupportedReasonKind::missing_metadata, "CoerceViaIO argument is null",
                                        location + ".arg");
        }
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_Aggref: {
        const auto* agg = reinterpret_cast<const Aggref*>(expr);
        const auto aggregateSupported = isAggregateSupported(agg);
        if (!aggregateSupported) {
            const auto functionName = postgresFunctionName(agg->aggfnoid);
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_function,
                                        functionName.empty()
                                            ? "unsupported aggregate function OID " + std::to_string(agg->aggfnoid)
                                            : "unsupported aggregate function " + functionName + "()",
                                        location);
        }
        mergeAnalyzerResult(result, analyzeTargetList(agg->args, location + ".args"));
        if (!aggregateSupported || agg->aggtype != BYTEAOID) {
            mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        }
        return supportedOrUnsupported(result);
    }

    case T_NullTest: {
        const auto* nullTest = reinterpret_cast<const NullTest*>(expr);
        mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(nullTest->arg), location + ".arg"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_BooleanTest:
        return AnalyzerResult::unsupported(UnsupportedReasonKind::unsupported_expr_node,
                                           "unsupported expression node BooleanTest", location);

    default:
        return AnalyzerResult::unsupported(UnsupportedReasonKind::unsupported_expr_node,
                                           "unsupported expression node tag " + std::to_string(nodeTag(expr)), location);
    }
}

auto QueryAnalyzer::checkCommandType(const PlannedStmt* stmt) -> bool {
    return stmt != nullptr && stmt->commandType == CMD_SELECT;
}

auto QueryAnalyzer::isTypeSupportedByMLIR(const Oid postgresType) -> bool {
    return postgresTypeIsMLIRSupported(postgresType);
}

auto QueryAnalyzer::isFunctionSupported(const FuncExpr* func) -> bool {
    return functionExprMatchesAny(func, supportedScalarFunctions, std::size(supportedScalarFunctions));
}

auto QueryAnalyzer::isAggregateSupported(const Aggref* agg) -> bool {
    if (!agg) {
        return false;
    }
    return catalogFunctionMatchesAny(agg->aggfnoid, supportedAggregates, std::size(supportedAggregates));
}

auto QueryAnalyzer::isOperatorSupported(const OpExpr* op) -> bool {
    return operatorExprMatchesCatalog(op) && operatorSignatureIsLowerable(op);
}

auto QueryAnalyzer::isCollationSupported(const Oid collationOid) -> bool {
    return postgresCollationIsSupported(collationOid);
}

auto QueryAnalyzer::analyzeNodeForTesting(const Plan* plan) -> AnalyzerResult {
    return analyzeNode(plan, "Plan");
}

auto QueryAnalyzer::analyzeExprForTesting(const Node* expr) -> AnalyzerResult {
    return analyzeExpr(expr, "Expr");
}

auto QueryAnalyzer::logExecutionTree(Plan* rootPlan) -> void {
    if (!rootPlan) {
        return;
    }
    PGX_LOG(AST_TRANSLATE, DEBUG, "=== POSTGRESQL EXECUTION TREE ===");

    char* plan_str = nodeToString(rootPlan);
    char* pretty_str = pretty_format_node_dump(plan_str);

    PGX_LOG(AST_TRANSLATE, DEBUG, "\n%s", pretty_str);

    pfree(pretty_str);
    pfree(plan_str);
    PGX_LOG(AST_TRANSLATE, TRACE, "=== END EXECUTION TREE ===");
}

auto QueryAnalyzer::validateAndLogPlanStructure(const PlannedStmt* stmt) -> bool {
    const auto rootPlan = stmt->planTree;
    Plan* scanPlan = nullptr;

    logExecutionTree(rootPlan);

    if (stmt->subplans && list_length(stmt->subplans) > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "=== SUBPLANS (%d total) ===", list_length(stmt->subplans));

        int i = 1;
        ListCell* lc = nullptr;
        foreach (lc, stmt->subplans) {
            Plan* subplan = (Plan*)lfirst(lc);
            PGX_LOG(AST_TRANSLATE, DEBUG, "\n--- SubPlan %d ---", i);

            char* plan_str = nodeToString(subplan);
            char* pretty_str = pretty_format_node_dump(plan_str);
            PGX_LOG(AST_TRANSLATE, DEBUG, "\n%s", pretty_str);

            pfree(pretty_str);
            pfree(plan_str);
            i++;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "=== END SUBPLANS ===\n");
    }
    if (rootPlan->type == T_SeqScan) {
        // Pattern 1: Simple table scan
        scanPlan = rootPlan;
        PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Simple SeqScan query");
    } else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        // Pattern 2: Aggregation with SeqScan
        scanPlan = rootPlan->lefttree;
        PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Aggregate query with SeqScan source");
    } else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_Gather) {
        // Pattern 3: Parallel aggregation (Agg  Gather  Agg  SeqScan)
        auto* gatherPlan = rootPlan->lefttree;
        if (gatherPlan->lefttree && gatherPlan->lefttree->type == T_Agg) {
            auto* innerAggPlan = gatherPlan->lefttree;
            if (innerAggPlan->lefttree && innerAggPlan->lefttree->type == T_SeqScan) {
                scanPlan = innerAggPlan->lefttree;
                PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Parallel aggregate query (AggGatherAggSeqScan)");
            }
        }

        if (!scanPlan) {
            PGX_LOG(AST_TRANSLATE, DEBUG, " PARTIAL SUPPORT: Gather pattern recognized but structure unexpected");
            // Still accept it for now to allow testing
        }
    } else {
        // TODO: NV haha this should be a warning, but it triggers so many integration tests... really makes you
        // wonder what's the point of this file...
        // Accept unknown patterns for comprehensive testing
        PGX_LOG(AST_TRANSLATE, DEBUG, " UNKNOWN PATTERN: Accepting for testing but may need implementation");
    }

    if (scanPlan) {
        const auto scan = reinterpret_cast<SeqScan*>(scanPlan);
        const auto rte = static_cast<RangeTblEntry*>(list_nth(stmt->rtable, scan->scan.scanrelid - 1));

        PGX_LOG(AST_TRANSLATE, DEBUG, " Table OID: %d", rte->relid);
        g_jit_table_oid = rte->relid;
        PGX_LOG(AST_TRANSLATE, DEBUG, " Set g_jit_table_oid to: %d", g_jit_table_oid);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " No scan plan extracted - query may not access tables directly");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, " QUERY ACCEPTED: Proceeding to MLIR compilation pipeline");
    return true;
}

#endif // POSTGRESQL_EXTENSION

} // namespace pgx_lower
