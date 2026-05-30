#include "pgx-lower/frontend/SQL/query_analyzer.h"

#include "pgx_lower_constants.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
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

    const auto COMPATIBLE = isSelectStatement && hasCompatibleTypes
                            && (requiresSeqScan || requiresAggregation || requiresJoin || requiresLimit);
    if (COMPATIBLE) {
        PGX_LOG(AST_TRANSLATE, DEBUG, " MLIR COMPATIBLE: Query accepted for compilation");
        return true;
    }         if (!isSelectStatement) {
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

auto QueryAnalyzer::analyzePlan(const PlannedStmt* stmt) -> QueryCapabilities {
    auto caps = QueryCapabilities{};

    if (!stmt || !stmt->planTree) {
        const auto error = ErrorManager::queryAnalysisError("No plan tree to analyze");
        ErrorManager::reportError(error);
        return caps;
    }

    try {
        caps.isSelectStatement = checkCommandType(stmt);
        if (!caps.isSelectStatement) {
            return caps;
}

        caps = analyzeNode(stmt->planTree);
        caps.isSelectStatement = true; // Preserve the SELECT check
        analyzeTypes(stmt->planTree, caps);

        return caps;
    } catch (const std::exception& e) {
        const auto error = ErrorManager::queryAnalysisError("Exception during plan analysis: " + std::string(e.what()));
        ErrorManager::reportError(error);
        return caps;
    }
}

auto QueryAnalyzer::analyzeNode(const Plan* plan) -> QueryCapabilities {
    auto caps = QueryCapabilities{};

    if (!plan) {
        return caps;
    }

    switch (nodeTag(plan)) {
    case T_SeqScan: analyzeSeqScan(reinterpret_cast<const SeqScan*>(plan), caps); break;

    case T_IndexScan:
    case T_IndexOnlyScan:
    case T_BitmapHeapScan: caps.requiresSeqScan = true; break;

    case T_NestLoop:
    case T_MergeJoin:
    case T_HashJoin: caps.requiresJoin = true; break;

    case T_Sort:
        // TODO: NV: This permits sort nodes with expressions in them. It isn't supposed to, and they just crash.
        //           They should be disabled here because lingodb doesn't support them either.
        caps.requiresSort = true;
        break;

    case T_Limit: caps.requiresLimit = true; break;

    case T_Agg: caps.requiresAggregation = true; break;

    case T_SubqueryScan:
        {
            const auto* subquery_scan = reinterpret_cast<const SubqueryScan*>(plan);
            if (subquery_scan->subplan) {
                const auto SUB_CAPS = analyzeNode(subquery_scan->subplan);
                caps.requiresSeqScan |= SUB_CAPS.requiresSeqScan;
                caps.requiresFilter |= SUB_CAPS.requiresFilter;
                caps.requiresProjection |= SUB_CAPS.requiresProjection;
                caps.requiresAggregation |= SUB_CAPS.requiresAggregation;
                caps.requiresJoin |= SUB_CAPS.requiresJoin;
                caps.requiresSort |= SUB_CAPS.requiresSort;
                caps.requiresLimit |= SUB_CAPS.requiresLimit;
                PGX_LOG(AST_TRANSLATE, DEBUG, "SubqueryScan propagating capabilities from subplan");
            }
        }
        break;

    case T_Result:
    case T_Material:
    case T_Hash:
    case T_Unique:
    case T_SetOp:
    case T_Group: PGX_LOG(AST_TRANSLATE, DEBUG, "Accepting node type %d for MLIR compilation", nodeTag(plan)); break;

    default:
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unknown node type %d - accepting for MLIR compilation", nodeTag(plan));
        break;
    }

    analyzeFilter(plan, caps);
    analyzeProjection(plan, caps);
    if (plan->lefttree) {
        const auto LEFT_CAPS = analyzeNode(plan->lefttree);
        caps.requiresSeqScan |= LEFT_CAPS.requiresSeqScan;
        caps.requiresFilter |= LEFT_CAPS.requiresFilter;
        caps.requiresProjection |= LEFT_CAPS.requiresProjection;
        caps.requiresAggregation |= LEFT_CAPS.requiresAggregation;
        caps.requiresJoin |= LEFT_CAPS.requiresJoin;
        caps.requiresSort |= LEFT_CAPS.requiresSort;
        caps.requiresLimit |= LEFT_CAPS.requiresLimit;
    }

    if (plan->righttree) {
        const auto RIGHT_CAPS = analyzeNode(plan->righttree);
        caps.requiresSeqScan |= RIGHT_CAPS.requiresSeqScan;
        caps.requiresFilter |= RIGHT_CAPS.requiresFilter;
        caps.requiresProjection |= RIGHT_CAPS.requiresProjection;
        caps.requiresAggregation |= RIGHT_CAPS.requiresAggregation;
        caps.requiresJoin |= RIGHT_CAPS.requiresJoin;
        caps.requiresSort |= RIGHT_CAPS.requiresSort;
        caps.requiresLimit |= RIGHT_CAPS.requiresLimit;
    }

    return caps;
}

auto QueryAnalyzer::analyzeSeqScan(const SeqScan* seqScan, QueryCapabilities& caps) -> void {
    caps.requiresSeqScan = true;
}

auto QueryAnalyzer::analyzeFilter(const Plan* plan, QueryCapabilities& caps) -> void {
    if (plan->qual) {
        caps.requiresFilter = true;
    }
}

auto QueryAnalyzer::analyzeProjection(const Plan*  /*plan*/, QueryCapabilities&  /*caps*/) -> void {
}

auto QueryAnalyzer::analyzeTypes(const Plan* plan, QueryCapabilities& caps) -> void {
    if (!plan || !plan->targetlist) {
        caps.hasCompatibleTypes = false;
        PGX_ERROR("don't pass in a nullable plan thanks");
        throw std::runtime_error("don't pass in a nullable plan thanks");
    }

    auto column_types = std::vector<Oid>{};
    ListCell* lc = nullptr;

    // Extract types from plan's target list
    foreach (lc, plan->targetlist) {
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle && !tle->resjunk && tle->expr) {
            // Check if this is a computed expression (not just a simple Var)
            if (nodeTag(tle->expr) != T_Var) {
                caps.hasExpressions = true;
            }

            // Later we can add more sophisticated filtering
            if (IsA(tle->expr, FuncExpr)) {
                const auto* func_expr = reinterpret_cast<FuncExpr*>(tle->expr);
                char* const func_name = get_func_name(func_expr->funcid);
                if (func_name) {
                    std::string const func(func_name);
                    pfree(func_name);
                    if (func == "upper" || func == "lower" || func == "substring" || func == "varchar" || func == "text"
                        || func == "char" || func == "bpchar" || func == "int4" || func == "int8" || func == "numeric"
                        || func == "float4" || func == "float8")
                    {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Supported function in targetlist: %s", func.c_str());
                    } else {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Unsupported function in targetlist: %s", func.c_str());
                        caps.hasCompatibleTypes = false;
                        return;
                    }
                } else {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Unknown function in targetlist: %d", func_expr->funcid);
                    caps.hasCompatibleTypes = false;
                    return;
                }
            }

            Oid const column_type = exprType(reinterpret_cast<Node*>(tle->expr));
            column_types.push_back(column_type);
        }
    }

    if (column_types.empty()) {
        caps.hasCompatibleTypes = false;
        return;
    }

    auto [supportedCount, unsupportedCount] = analyzeTypeCompatibility(column_types);
    caps.hasCompatibleTypes = (unsupportedCount == 0);
}

auto QueryAnalyzer::checkCommandType(const PlannedStmt* stmt) -> bool {
    if (!stmt) {
        PGX_ERROR("don't pass in a nullable stmt thanks");
        throw std::runtime_error("don't pass in a nullable stmt thanks");
    }
    return (stmt->commandType == CMD_SELECT);
}

auto QueryAnalyzer::isTypeSupportedByMLIR(const Oid POSTGRES_TYPE) -> bool {
    switch (POSTGRES_TYPE) {
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

auto QueryAnalyzer::analyzeTypeCompatibility(const std::vector<Oid>& types) -> std::pair<int, int> {
    auto supported_count = 0;
    auto unsupported_count = 0;

    for (const auto TYPE : types) {
        if (isTypeSupportedByMLIR(TYPE)) {
            supported_count++;
        } else {
            unsupported_count++;
}
    }

    return {supported_count, unsupported_count};
}

auto QueryAnalyzer::logExecutionTree(Plan* root_plan) -> void {
    if (!root_plan) {
        return;
}
    PGX_LOG(AST_TRANSLATE, DEBUG, "=== POSTGRESQL EXECUTION TREE ===");

    char* const plan_str = nodeToString(root_plan);
    char* const pretty_str = pretty_format_node_dump(plan_str);

    PGX_LOG(AST_TRANSLATE, DEBUG, "\n%s", pretty_str);

    pfree(pretty_str);
    pfree(plan_str);
    PGX_LOG(AST_TRANSLATE, TRACE, "=== END EXECUTION TREE ===");
}

auto QueryAnalyzer::validateAndLogPlanStructure(const PlannedStmt* stmt) -> bool {
    auto *const ROOT_PLAN = stmt->planTree;
    Plan* scan_plan = nullptr;

    logExecutionTree(ROOT_PLAN);

    if (stmt->subplans && list_length(stmt->subplans) > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "=== SUBPLANS (%d total) ===", list_length(stmt->subplans));

        int i = 1;
        ListCell* lc = nullptr;
        foreach(lc, stmt->subplans) {
            Plan* const subplan = (Plan*)lfirst(lc);
            PGX_LOG(AST_TRANSLATE, DEBUG, "\n--- SubPlan %d ---", i);

            char* const plan_str = nodeToString(subplan);
            char* const pretty_str = pretty_format_node_dump(plan_str);
            PGX_LOG(AST_TRANSLATE, DEBUG, "\n%s", pretty_str);

            pfree(pretty_str);
            pfree(plan_str);
            i++;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "=== END SUBPLANS ===\n");
    }
    if (ROOT_PLAN->type == T_SeqScan) {
        // Pattern 1: Simple table scan
        scan_plan = ROOT_PLAN;
        PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Simple SeqScan query");
    } else if (ROOT_PLAN->type == T_Agg && ROOT_PLAN->lefttree && ROOT_PLAN->lefttree->type == T_SeqScan) {
        // Pattern 2: Aggregation with SeqScan
        scan_plan = ROOT_PLAN->lefttree;
        PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Aggregate query with SeqScan source");
    } else if (ROOT_PLAN->type == T_Agg && ROOT_PLAN->lefttree && ROOT_PLAN->lefttree->type == T_Gather) {
        // Pattern 3: Parallel aggregation (Agg  Gather  Agg  SeqScan)
        auto* gather_plan = ROOT_PLAN->lefttree;
        if (gather_plan->lefttree && gather_plan->lefttree->type == T_Agg) {
            auto* inner_agg_plan = gather_plan->lefttree;
            if (inner_agg_plan->lefttree && inner_agg_plan->lefttree->type == T_SeqScan) {
                scan_plan = inner_agg_plan->lefttree;
                PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Parallel aggregate query (AggGatherAggSeqScan)");
            }
        }

        if (!scan_plan) {
            PGX_LOG(AST_TRANSLATE, DEBUG, " PARTIAL SUPPORT: Gather pattern recognized but structure unexpected");
            // Still accept it for now to allow testing
        }
    } else {
        // TODO: NV haha this should be a warning, but it triggers so many integration tests... really makes you
        // wonder what's the point of this file...
        // Accept unknown patterns for comprehensive testing
        PGX_LOG(AST_TRANSLATE, DEBUG, " UNKNOWN PATTERN: Accepting for testing but may need implementation");
    }

    if (scan_plan) {
        auto *const SCAN = reinterpret_cast<SeqScan*>(scan_plan);
        auto *const RTE = static_cast<RangeTblEntry*>(list_nth(stmt->rtable, SCAN->scan.scanrelid - 1));

        PGX_LOG(AST_TRANSLATE, DEBUG, " Table OID: %d", RTE->relid);
        g_jit_table_oid = RTE->relid;
        PGX_LOG(AST_TRANSLATE, DEBUG, " Set g_jit_table_oid to: %d", g_jit_table_oid);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " No scan plan extracted - query may not access tables directly");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, " QUERY ACCEPTED: Proceeding to MLIR compilation pipeline");
    return true;
}

#endif // POSTGRESQL_EXTENSION

auto QueryAnalyzer::analyzeForTesting(const char* query_text) -> QueryCapabilities {
    auto caps = QueryCapabilities{};

    if (!query_text) {
        return caps;
}

    if ((strstr(query_text, "SELECT") != nullptr) && (strstr(query_text, "FROM") != nullptr)) {
        caps.isSelectStatement = true;
        caps.requiresSeqScan = true;
        caps.hasCompatibleTypes = true;
    }

    // Check for projection (specific columns rather than *)
    // TODO: NV: Errr... yeah... hmm... this looks sus. TODO: Delete this entire method!
    if ((strstr(query_text, "SELECT") != nullptr) && (strstr(query_text, "SELECT *") == nullptr)) {
        const char* const select_pos = strstr(query_text, "SELECT");
        const char* const from_pos = strstr(query_text, "FROM");
        if (select_pos && from_pos) {
            const char* select_content = select_pos + 6; // "select"
            while (*select_content == ' ') {
                select_content++;
            }
            if (select_content < from_pos && *select_content != '*') {
                caps.requiresProjection = true;
            }
        }
    }

    if (strstr(query_text, "WHERE") != nullptr) {
        caps.requiresFilter = true;
}
    if (strstr(query_text, "JOIN") != nullptr) {
        caps.requiresJoin = true;
}
    if (strstr(query_text, "ORDER BY") != nullptr) {
        caps.requiresSort = true;
}
    if (strstr(query_text, "LIMIT") != nullptr) {
        caps.requiresLimit = true;
}
    if ((strstr(query_text, "COUNT") != nullptr) || (strstr(query_text, "SUM") != nullptr)
        || (strstr(query_text, "AVG") != nullptr) || (strstr(query_text, "GROUP BY") != nullptr)) {
        caps.requiresAggregation = true;
}
    if (strstr(query_text, "(SELECT") != nullptr) {
        caps.requiresJoin = true; // Treat nested queries as requiring joins for now
}

    return caps;
}

} // namespace pgx_lower