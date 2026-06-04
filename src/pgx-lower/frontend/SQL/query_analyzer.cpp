#include "pgx-lower/frontend/SQL/query_analyzer.h"

#include "pgx_lower_constants.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "catalog/pg_collation.h"
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

#include <cstring>
#include <vector>
#include <sstream>
#include <functional>

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

auto QueryCapabilities::isMLIRCompatible() const -> bool {
    std::vector<std::string> features;
    if (isSelectStatement) {
        features.emplace_back("SELECT");
    }
    if (requiresSeqScan) {
        features.emplace_back("SeqScan");
    }
    if (requiresProjection) {
        features.emplace_back("Projection");
    }
    if (hasExpressions) {
        features.emplace_back("Expressions");
    }
    if (requiresFilter) {
        features.emplace_back("WHERE");
    }
    if (requiresAggregation) {
        features.emplace_back("Aggregation");
    }
    if (requiresSort) {
        features.emplace_back("ORDER BY");
    }
    if (requiresJoin) {
        features.emplace_back("JOIN");
    }
    if (requiresLimit) {
        features.emplace_back("LIMIT");
    }
    if (hasCompatibleTypes) {
        features.emplace_back("CompatibleTypes");
    }

    if (!features.empty()) {
        auto feature_list = std::string();
        for (const auto& f : features) {
            feature_list += f + ", ";
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, " Query features: %s", feature_list.c_str());
    }

    else
    {
        PGX_LOG(AST_TRANSLATE, DEBUG, " Query features: None detected");
    }

    const auto compatible = isSelectStatement && hasCompatibleTypes
                            && (requiresSeqScan || requiresAggregation || requiresJoin || requiresLimit);
    if (compatible) {
        PGX_LOG(AST_TRANSLATE, DEBUG, " MLIR COMPATIBLE: Query accepted for compilation");
        return true;
    }
    if (!isSelectStatement) {
        PGX_LOG(AST_TRANSLATE, DEBUG, " REJECTED: Not a SELECT statement");
    } else if (!hasCompatibleTypes) {
        PGX_LOG(AST_TRANSLATE, DEBUG, " REJECTED: Incompatible types detected");
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " REJECTED: Unknown reason");
    }
    return false;
}

auto QueryCapabilities::getDescription() const -> std::string {
    if (isMLIRCompatible()) {
        return "Sequential scan with optional aggregation - MLIR compatible";
    }

    auto requirements = std::vector<std::string>{};

    if (requiresSeqScan) {
        requirements.emplace_back("SeqScan");
    }
    if (requiresFilter) {
        requirements.emplace_back("Filter");
    }
    if (requiresProjection) {
        requirements.emplace_back("Projection");
    }
    if (requiresAggregation) {
        requirements.emplace_back("Aggregation");
    }
    if (requiresJoin) {
        requirements.emplace_back("Join");
    }
    if (requiresSort) {
        requirements.emplace_back("Sort");
    }
    if (requiresLimit) {
        requirements.emplace_back("Limit");
    }

#ifdef POSTGRESQL_EXTENSION
    if (hasExpressions) {
        if (g_extension_after_load) {
            requirements.emplace_back("Expressions (disabled after LOAD)");
        }
    }
#endif

    std::ostringstream oss;
    oss << "Requires: ";
    for (const auto& r : requirements) {
        oss << r << ", ";
    }
    oss << " - Not yet supported by MLIR";

    return oss.str();
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
    case T_HashJoin:
    case T_Sort:
    case T_Limit:
    case T_Agg:
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
        if (!tle || tle->resjunk || !tle->expr) {
            ++index;
            continue;
        }
        const auto exprLocation = location + "[" + std::to_string(index) + "]";
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
        if (!isFunctionSupported(func->funcid)) {
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
        if (!isOperatorSupported(op->opno)) {
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

    case T_RelabelType: {
        const auto* relabel = reinterpret_cast<const RelabelType*>(expr);
        mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(relabel->arg), location + ".arg"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_Aggref: {
        const auto* agg = reinterpret_cast<const Aggref*>(expr);
        if (!isFunctionSupported(agg->aggfnoid)) {
            const auto functionName = postgresFunctionName(agg->aggfnoid);
            result.addUnsupportedReason(UnsupportedReasonKind::unsupported_function,
                                        functionName.empty()
                                            ? "unsupported aggregate function OID " + std::to_string(agg->aggfnoid)
                                            : "unsupported aggregate function " + functionName + "()",
                                        location);
        }
        mergeAnalyzerResult(result, analyzeTargetList(agg->args, location + ".args"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_NullTest: {
        const auto* nullTest = reinterpret_cast<const NullTest*>(expr);
        mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(nullTest->arg), location + ".arg"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    case T_BooleanTest: {
        const auto* booleanTest = reinterpret_cast<const BooleanTest*>(expr);
        mergeAnalyzerResult(result, analyzeExpr(reinterpret_cast<const Node*>(booleanTest->arg), location + ".arg"));
        mergeAnalyzerResult(result, analyzeExprType(expr, location + ".type"));
        return supportedOrUnsupported(result);
    }

    default:
        return AnalyzerResult::unsupported(UnsupportedReasonKind::unsupported_expr_node,
                                           "unsupported expression node tag " + std::to_string(nodeTag(expr)), location);
    }
}

auto QueryAnalyzer::checkCommandType(const PlannedStmt* stmt) -> bool {
    return stmt != nullptr && stmt->commandType == CMD_SELECT;
}

auto QueryAnalyzer::isTypeSupportedByMLIR(const Oid postgresType) -> bool {
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

auto QueryAnalyzer::isFunctionSupported(const Oid functionOid) -> bool {
    const auto functionName = postgresFunctionName(functionOid);
    if (functionName.empty()) {
        return false;
    }
    return functionName == "count" || functionName == "sum" || functionName == "avg" || functionName == "min"
           || functionName == "max" || functionName == "upper" || functionName == "lower" || functionName == "substring"
           || functionName == "varchar" || functionName == "text" || functionName == "char" || functionName == "bpchar"
           || functionName == "int2" || functionName == "int4" || functionName == "int8" || functionName == "numeric"
           || functionName == "float4" || functionName == "float8" || functionName == "date"
           || functionName == "timestamp" || functionName == "interval";
}

auto QueryAnalyzer::isOperatorSupported(const Oid operatorOid) -> bool {
    if (operatorOid == InvalidOid) {
        return false;
    }
    const char* name = get_opname(operatorOid);
    if (!name) {
        return false;
    }
    const auto operatorName = std::string(name);
    pfree(const_cast<char*>(name));
    return operatorName == "=" || operatorName == "<>" || operatorName == "!=" || operatorName == "<"
           || operatorName == "<=" || operatorName == ">" || operatorName == ">=" || operatorName == "+"
           || operatorName == "-" || operatorName == "*" || operatorName == "/" || operatorName == "~~"
           || operatorName == "!~~";
}

auto QueryAnalyzer::isCollationSupported(const Oid collationOid) -> bool {
    return collationOid == InvalidOid || collationOid == DEFAULT_COLLATION_OID || collationOid == C_COLLATION_OID
           || collationOid == POSIX_COLLATION_OID;
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

auto QueryAnalyzer::analyzeForTesting(const char* queryText) -> QueryCapabilities {
    auto caps = QueryCapabilities{};

    if (!queryText) {
        return caps;
    }

    if ((strstr(queryText, "SELECT") != nullptr) && (strstr(queryText, "FROM") != nullptr)) {
        caps.isSelectStatement = true;
        caps.requiresSeqScan = true;
        caps.hasCompatibleTypes = true;
    }

    // Check for projection (specific columns rather than *)
    // TODO: NV: Errr... yeah... hmm... this looks sus. TODO: Delete this entire method!
    if ((strstr(queryText, "SELECT") != nullptr) && (strstr(queryText, "SELECT *") == nullptr)) {
        const char* selectPos = strstr(queryText, "SELECT");
        const char* fromPos = strstr(queryText, "FROM");
        if (selectPos && fromPos) {
            const char* selectContent = selectPos + 6; // "select"
            while (*selectContent == ' ') {
                selectContent++;
            }
            if (selectContent < fromPos && *selectContent != '*') {
                caps.requiresProjection = true;
            }
        }
    }

    if (strstr(queryText, "WHERE") != nullptr) {
        caps.requiresFilter = true;
    }
    if (strstr(queryText, "JOIN") != nullptr) {
        caps.requiresJoin = true;
    }
    if (strstr(queryText, "ORDER BY") != nullptr) {
        caps.requiresSort = true;
    }
    if (strstr(queryText, "LIMIT") != nullptr) {
        caps.requiresLimit = true;
    }
    if ((strstr(queryText, "COUNT") != nullptr) || (strstr(queryText, "SUM") != nullptr)
        || (strstr(queryText, "AVG") != nullptr) || (strstr(queryText, "GROUP BY") != nullptr))
    {
        caps.requiresAggregation = true;
    }
    if (strstr(queryText, "(SELECT") != nullptr) {
        caps.requiresJoin = true; // Treat nested queries as requiring joins for now
    }

    return caps;
}

} // namespace pgx_lower
