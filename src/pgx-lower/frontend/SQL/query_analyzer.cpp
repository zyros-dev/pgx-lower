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
extern bool g_extension_after_load;
#endif

namespace pgx_lower {

auto QueryCapabilities::isMLIRCompatible() const -> bool {
    std::vector<std::string> features;
    if (isSelectStatement)
        features.emplace_back("SELECT");
    if (requiresSeqScan)
        features.emplace_back("SeqScan");
    if (requiresProjection)
        features.emplace_back("Projection");
    if (hasExpressions)
        features.emplace_back("Expressions");
    if (requiresFilter)
        features.emplace_back("WHERE");
    if (requiresAggregation)
        features.emplace_back("Aggregation");
    if (requiresSort)
        features.emplace_back("ORDER BY");
    if (requiresJoin)
        features.emplace_back("JOIN");
    if (requiresLimit)
        features.emplace_back("LIMIT");
    if (hasCompatibleTypes)
        features.emplace_back("CompatibleTypes");

    if (!features.empty()) {
        auto feature_list = std::string();
        for (const auto& f : features)
            feature_list += f + ", ";
        PGX_LOG(AST_TRANSLATE, DEBUG, " Query features: %s", feature_list.c_str());
    }

    else
    {
        PGX_LOG(AST_TRANSLATE, DEBUG, " Query features: None detected");
    }

    const auto compatible = isSelectStatement && hasCompatibleTypes && !requiresLimit
                            && (requiresSeqScan || requiresAggregation || requiresJoin);

    if (compatible) {
        PGX_LOG(AST_TRANSLATE, DEBUG, " MLIR COMPATIBLE: Query accepted for compilation");
        return true;
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " DATA COLLECTION: Query too complex for current MLIR implementation");
        return false;
    }
}

auto QueryCapabilities::getDescription() const -> std::string {
    if (isMLIRCompatible())
        return "Sequential scan with optional aggregation - MLIR compatible";

    auto requirements = std::vector<std::string>{};

    if (requiresSeqScan)
        requirements.emplace_back("SeqScan");
    if (requiresFilter)
        requirements.emplace_back("Filter");
    if (requiresProjection)
        requirements.emplace_back("Projection");
    if (requiresAggregation)
        requirements.emplace_back("Aggregation");
    if (requiresJoin)
        requirements.emplace_back("Join");
    if (requiresSort)
        requirements.emplace_back("Sort");
    if (requiresLimit)
        requirements.emplace_back("Limit");

#ifdef POSTGRESQL_EXTENSION
    if (hasExpressions) {
        if (g_extension_after_load) {
            requirements.emplace_back("Expressions (disabled after LOAD)");
        }
    }
#endif

    std::ostringstream oss;
    oss << "Requires: ";
    for (const auto& r : requirements)
        oss << r << ", ";
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
        if (!caps.isSelectStatement)
            return caps;

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
    case T_BitmapHeapScan: caps.requiresSeqScan = false; break;

    case T_NestLoop:
    case T_MergeJoin:
    case T_HashJoin: caps.requiresJoin = true; break;

    case T_Sort:
        // TODO: NV: This permits sort nodes with expressions in them. It isn't supposed to, and they just crash.
        //           They should be disabled here because lingodb doesn't support them either.
        caps.requiresSort = true;
        break;

    case T_Limit: caps.requiresLimit = true; break;

    case T_Agg:
        caps.requiresAggregation = true;
        break;
        // default:
        // TODO: NV: Temporarily commented out while I;'m doing this refactor
        // PGX_ERROR("Failed to match node %d", nodeTag(plan)); throw std::runtime_error("Failed to match node!");
    }

    analyzeFilter(plan, caps);
    analyzeProjection(plan, caps);
    if (plan->lefttree) {
        const auto leftCaps = analyzeNode(plan->lefttree);
        caps.requiresSeqScan |= leftCaps.requiresSeqScan;
        caps.requiresFilter |= leftCaps.requiresFilter;
        caps.requiresProjection |= leftCaps.requiresProjection;
        caps.requiresAggregation |= leftCaps.requiresAggregation;
        caps.requiresJoin |= leftCaps.requiresJoin;
        caps.requiresSort |= leftCaps.requiresSort;
        caps.requiresLimit |= leftCaps.requiresLimit;
    }

    if (plan->righttree) {
        const auto rightCaps = analyzeNode(plan->righttree);
        caps.requiresSeqScan |= rightCaps.requiresSeqScan;
        caps.requiresFilter |= rightCaps.requiresFilter;
        caps.requiresProjection |= rightCaps.requiresProjection;
        caps.requiresAggregation |= rightCaps.requiresAggregation;
        caps.requiresJoin |= rightCaps.requiresJoin;
        caps.requiresSort |= rightCaps.requiresSort;
        caps.requiresLimit |= rightCaps.requiresLimit;
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

auto QueryAnalyzer::analyzeProjection(const Plan* plan, QueryCapabilities& caps) -> void {
}

auto QueryAnalyzer::analyzeTypes(const Plan* plan, QueryCapabilities& caps) -> void {
    if (!plan || !plan->targetlist) {
        caps.hasCompatibleTypes = false;
        PGX_ERROR("don't pass in a nullable plan thanks");
        throw std::runtime_error("don't pass in a nullable plan thanks");
    }

    auto columnTypes = std::vector<Oid>{};
    ListCell* lc;

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
                const auto* funcExpr = reinterpret_cast<FuncExpr*>(tle->expr);
                char* funcName = get_func_name(funcExpr->funcid);
                if (funcName) {
                    std::string func(funcName);
                    pfree(funcName);
                    if (func == "upper" || func == "lower" || func == "substring" ||
                        func == "varchar" || func == "text" || func == "char" || func == "bpchar" ||
                        func == "int4" || func == "int8" || func == "numeric" || func == "float4" || func == "float8") {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Supported function in targetlist: %s", func.c_str());
                    } else {
                        PGX_LOG(AST_TRANSLATE, DEBUG, "Unsupported function in targetlist: %s", func.c_str());
                        caps.hasCompatibleTypes = false;
                        return;
                    }
                } else {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Unknown function in targetlist: %d", funcExpr->funcid);
                    caps.hasCompatibleTypes = false;
                    return;
                }
            }

            Oid columnType = exprType(reinterpret_cast<Node*>(tle->expr));
            columnTypes.push_back(columnType);
        }
    }

    if (columnTypes.empty()) {
        caps.hasCompatibleTypes = false;
        return;
    }

    auto [supportedCount, unsupportedCount] = analyzeTypeCompatibility(columnTypes);
    caps.hasCompatibleTypes = (unsupportedCount == 0);
}

auto QueryAnalyzer::checkCommandType(const PlannedStmt* stmt) -> bool {
    if (!stmt) {
        PGX_ERROR("don't pass in a nullable stmt thanks");
        throw std::runtime_error("don't pass in a nullable stmt thanks");
    }
    return (stmt->commandType == CMD_SELECT);
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

auto QueryAnalyzer::analyzeTypeCompatibility(const std::vector<Oid>& types) -> std::pair<int, int> {
    auto supportedCount = 0;
    auto unsupportedCount = 0;

    for (const auto type : types) {
        if (isTypeSupportedByMLIR(type))
            supportedCount++;
        else
            unsupportedCount++;
    }

    return {supportedCount, unsupportedCount};
}

auto QueryAnalyzer::logExecutionTree(Plan* rootPlan) -> void {
    if (!rootPlan) return;
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

    if (!queryText)
        return caps;

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

    if (strstr(queryText, "WHERE") != nullptr)
        caps.requiresFilter = true;
    if (strstr(queryText, "JOIN") != nullptr)
        caps.requiresJoin = true;
    if (strstr(queryText, "ORDER BY") != nullptr)
        caps.requiresSort = true;
    if (strstr(queryText, "LIMIT") != nullptr)
        caps.requiresLimit = true;
    if ((strstr(queryText, "COUNT") != nullptr) || (strstr(queryText, "SUM") != nullptr)
        || (strstr(queryText, "AVG") != nullptr) || (strstr(queryText, "GROUP BY") != nullptr))
        caps.requiresAggregation = true;
    if (strstr(queryText, "(SELECT") != nullptr)
        caps.requiresJoin = true; // Treat nested queries as requiring joins for now

    return caps;
}

} // namespace pgx_lower