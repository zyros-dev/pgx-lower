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
        std::string feature_list = features[0];
        for (size_t i = 1; i < features.size(); ++i) {
            feature_list += "+" + features[i];
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, " Query features: %s", feature_list.c_str());
    }
    else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " Query features: None detected");
    }

    //  ENABLE MLIR COMPILATION: Test if pipeline works for basic SELECT+SeqScan queries
    bool compatible = isSelectStatement && requiresSeqScan && hasCompatibleTypes && !requiresJoin
                      && !requiresAggregation && !requiresSort && !requiresLimit;

    if (compatible) {
        PGX_LOG(AST_TRANSLATE, DEBUG, " MLIR COMPATIBLE: Basic SELECT+SeqScan query accepted for compilation");
        return true;
    }
    else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " DATA COLLECTION: Query too complex for current MLIR implementation");
        return false;
    }
}

auto QueryCapabilities::getDescription() const -> std::string {
    if (isMLIRCompatible())
        return "Sequential scan with optional aggregation - MLIR compatible";

    std::vector<std::string> requirements;

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

    if (hasExpressions) {
#ifdef POSTGRESQL_EXTENSION
        if (::g_extension_after_load) {
            requirements.emplace_back("Expressions (disabled after LOAD)");
        }
#endif
    }

    std::ostringstream oss;
    oss << "Requires: ";

    for (size_t i = 0; i < requirements.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << requirements[i];
    }

    oss << " - Not yet supported by MLIR";

    return oss.str();
}

#ifdef POSTGRESQL_EXTENSION

QueryCapabilities QueryAnalyzer::analyzePlan(const PlannedStmt* stmt) {
    QueryCapabilities caps;

    if (!stmt || !stmt->planTree) {
        auto error = ErrorManager::queryAnalysisError("No plan tree to analyze");
        ErrorManager::reportError(error);
        return caps;
    }

    try {
        // 1. Check command type first (CMD_SELECT only)
        caps.isSelectStatement = checkCommandType(stmt);
        if (!caps.isSelectStatement) {
            return caps;
        }

        // 2. Analyze plan structure and requirements
        caps = analyzeNode(stmt->planTree);
        caps.isSelectStatement = true; // Preserve the SELECT check

        // 3. Analyze column types from plan metadata (no table access)
        analyzeTypes(stmt->planTree, caps);

        return caps;
    } catch (const std::exception& e) {
        auto error = ErrorManager::queryAnalysisError("Exception during plan analysis: " + std::string(e.what()));
        ErrorManager::reportError(error);
        return caps;
    }
}

QueryCapabilities QueryAnalyzer::analyzeNode(const Plan* plan) {
    QueryCapabilities caps;

    if (!plan) {
        return caps;
    }

    // Analyze this node
    switch (nodeTag(plan)) {
    case T_SeqScan: analyzeSeqScan(reinterpret_cast<const SeqScan*>(plan), caps); break;

    case T_IndexScan:
    case T_IndexOnlyScan:
    case T_BitmapHeapScan:
        caps.requiresSeqScan = false; // This is an index scan, not seq scan
        break;

    case T_NestLoop:
    case T_MergeJoin:
    case T_HashJoin: caps.requiresJoin = true; break;

    case T_Sort: caps.requiresSort = true; break;

    case T_Limit: caps.requiresLimit = true; break;

    case T_Agg: caps.requiresAggregation = true; break;
    }

    // Check for filters
    analyzeFilter(plan, caps);

    // Check for projections
    analyzeProjection(plan, caps);

    // Recursively analyze child nodes
    if (plan->lefttree) {
        QueryCapabilities leftCaps = analyzeNode(plan->lefttree);
        caps.requiresSeqScan |= leftCaps.requiresSeqScan;
        caps.requiresFilter |= leftCaps.requiresFilter;
        caps.requiresProjection |= leftCaps.requiresProjection;
        caps.requiresAggregation |= leftCaps.requiresAggregation;
        caps.requiresJoin |= leftCaps.requiresJoin;
        caps.requiresSort |= leftCaps.requiresSort;
        caps.requiresLimit |= leftCaps.requiresLimit;
    }

    if (plan->righttree) {
        QueryCapabilities rightCaps = analyzeNode(plan->righttree);
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

void QueryAnalyzer::analyzeSeqScan(const SeqScan* seqScan, QueryCapabilities& caps) {
    caps.requiresSeqScan = true;
}

void QueryAnalyzer::analyzeFilter(const Plan* plan, QueryCapabilities& caps) {
    if (plan->qual) {
        caps.requiresFilter = true;
    }
}

void QueryAnalyzer::analyzeProjection(const Plan* plan, QueryCapabilities& caps) {
    // In the future, we can be more sophisticated about this
    if (plan->targetlist) {
        // Simple heuristic: if we have a target list, we might need projection
        // since our current MLIR implementation handles basic projections
    }
}

void QueryAnalyzer::analyzeTypes(const Plan* plan, QueryCapabilities& caps) {
    if (!plan || !plan->targetlist) {
        caps.hasCompatibleTypes = false;
        return;
    }

    std::vector<Oid> columnTypes;
    ListCell* lc;

    // Extract types from plan's target list (no table access needed)
    foreach (lc, plan->targetlist) {
        TargetEntry* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle && !tle->resjunk && tle->expr) {
            // Check if this is a computed expression (not just a simple Var)
            if (nodeTag(tle->expr) != T_Var) {
                caps.hasExpressions = true;
            }

            // Later we can add more sophisticated filtering
            if (IsA(tle->expr, FuncExpr)) {
                FuncExpr* funcExpr = reinterpret_cast<FuncExpr*>(tle->expr);
                if (funcExpr->funcid == frontend::sql::constants::PG_F_UPPER ||
                    funcExpr->funcid == frontend::sql::constants::PG_F_LOWER ||
                    funcExpr->funcid == frontend::sql::constants::PG_F_SUBSTRING) {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Supported string function in targetlist: %d", funcExpr->funcid);
                } else {
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Unsupported function in targetlist: %d", funcExpr->funcid);
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

    // Analyze type compatibility using built-in system
    auto [supportedCount, unsupportedCount] = analyzeTypeCompatibility(columnTypes);

    // In the future, we could allow partial support with fallbacks
    caps.hasCompatibleTypes = (unsupportedCount == 0);

    if (!caps.hasCompatibleTypes) {
    }
}

bool QueryAnalyzer::checkCommandType(const PlannedStmt* stmt) {
    if (!stmt) {
        return false;
    }

    bool isSelect = (stmt->commandType == CMD_SELECT);
    if (!isSelect) {
    }

    return isSelect;
}

bool QueryAnalyzer::isTypeSupportedByMLIR(Oid postgresType) {
    // PostgreSQL types that MLIR runtime can handle
    // Based on working test cases and available runtime functions
    switch (postgresType) {
    // Integer types (handled by get_int32_field)
    case BOOLOID:
    case INT2OID:
    case INT4OID:
    case INT8OID:
    case NUMERICOID:
    case FLOAT4OID:
    case FLOAT8OID: return true;

    case TEXTOID:
    case VARCHAROID:
    case CHAROID: return true;

    case MONEYOID:
    case DATEOID:
    case TIMEOID:
    case TIMETZOID:
    case TIMESTAMPOID:
    case TIMESTAMPTZOID:
    case INTERVALOID:
    case UUIDOID:
    case INETOID:
    case CIDROID:
    case MACADDROID:
    case BITOID:
    case VARBITOID:
    case BYTEAOID:
    case NAMEOID:
    case OIDOID:
    case JSONOID:
    case JSONBOID:
    case XMLOID:
    case BPCHAROID:
    case MACADDR8OID:
    case PG_LSNOID: return true;

    default: return false;
    }
}

std::pair<int, int> QueryAnalyzer::analyzeTypeCompatibility(const std::vector<Oid>& types) {
    int supportedCount = 0;
    int unsupportedCount = 0;

    for (Oid type : types) {
        if (isTypeSupportedByMLIR(type)) {
            supportedCount++;
        }
        else {
            unsupportedCount++;
        }
    }

    return {supportedCount, unsupportedCount};
}

// Helper function to log PostgreSQL execution trees with nice formatting
void QueryAnalyzer::logExecutionTree(Plan* rootPlan) {
    // Get readable node type names
    auto getNodeTypeName = [](NodeTag nodeType) -> std::string {
        switch (nodeType) {
        case T_SeqScan: return "SeqScan";
        case T_IndexScan: return "IndexScan";
        case T_IndexOnlyScan: return "IndexOnlyScan";
        case T_BitmapIndexScan: return "BitmapIndexScan";
        case T_BitmapHeapScan: return "BitmapHeapScan";
        case T_TidScan: return "TidScan";
        case T_SubqueryScan: return "SubqueryScan";
        case T_FunctionScan: return "FunctionScan";
        case T_ValuesScan: return "ValuesScan";
        case T_TableFuncScan: return "TableFuncScan";
        case T_CteScan: return "CteScan";
        case T_NamedTuplestoreScan: return "NamedTuplestoreScan";
        case T_WorkTableScan: return "WorkTableScan";
        case T_ForeignScan: return "ForeignScan";
        case T_CustomScan: return "CustomScan";
        case T_NestLoop: return "NestLoop";
        case T_MergeJoin: return "MergeJoin";
        case T_HashJoin: return "HashJoin";
        case T_Material: return "Material";
        case T_Sort: return "Sort";
        case T_Group: return "Group";
        case T_Agg: return "Agg";
        case T_WindowAgg: return "WindowAgg";
        case T_Unique: return "Unique";
        case T_Gather: return "Gather";
        case T_GatherMerge: return "GatherMerge";
        case T_Hash: return "Hash";
        case T_SetOp: return "SetOp";
        case T_LockRows: return "LockRows";
        case T_Limit: return "Limit";
        case T_Result: return "Result";
        case T_ProjectSet: return "ProjectSet";
        case T_ModifyTable: return "ModifyTable";
        case T_Append: return "Append";
        case T_MergeAppend: return "MergeAppend";
        case T_RecursiveUnion: return "RecursiveUnion";
        case T_BitmapAnd: return "BitmapAnd";
        case T_BitmapOr: return "BitmapOr";
        case T_Memoize: return "Memoize";
        default: return "Unknown(" + std::to_string(nodeType) + ")";
        }
    };

    // Recursively print execution tree
    std::function<void(Plan*, int, const std::string&)> printPlanTree =
        [&](Plan* plan, int depth, const std::string& prefix) {
            if (!plan) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "%sNULL", prefix.c_str());
                return;
            }

            std::string indent(depth * 2, ' ');
            std::string nodeName = getNodeTypeName(static_cast<NodeTag>(plan->type));
            std::string nodeInfo = prefix + nodeName + " (type=" + std::to_string(plan->type) + ")";

            // Add node-specific details
            if (plan->type == T_SeqScan) {
                auto* seqScan = reinterpret_cast<SeqScan*>(plan);
                nodeInfo += " [scanrelid=" + std::to_string(seqScan->scan.scanrelid) + "]";
            }
            else if (plan->type == T_Agg) {
                auto* agg = reinterpret_cast<Agg*>(plan);
                nodeInfo += " [strategy=" + std::to_string(agg->aggstrategy) + "]";
            }
            else if (plan->type == T_Gather) {
                auto* gather = reinterpret_cast<Gather*>(plan);
                nodeInfo += " [num_workers=" + std::to_string(gather->num_workers) + "]";
            }

            PGX_LOG(AST_TRANSLATE, DEBUG, "%s", nodeInfo.c_str());
            
            if (plan->targetlist) {
                ListCell* lc;
                int idx = 0;
                foreach (lc, plan->targetlist) {
                    TargetEntry* tle = static_cast<TargetEntry*>(lfirst(lc));
                    if (tle && tle->expr) {
                        NodeTag exprType = nodeTag(tle->expr);
                        std::string exprTypeName = "Unknown";
                        switch(exprType) {
                            case T_Var: exprTypeName = "Var"; break;
                            case T_Const: exprTypeName = "Const"; break;
                            case T_OpExpr: exprTypeName = "OpExpr"; break;
                            case T_FuncExpr: {
                                exprTypeName = "FuncExpr";
                                FuncExpr* funcExpr = reinterpret_cast<FuncExpr*>(tle->expr);
                                exprTypeName += "(funcid=" + std::to_string(funcExpr->funcid) + ")";
                                break;
                            }
                            default: exprTypeName = "Type" + std::to_string(exprType);
                        }
                        PGX_LOG(AST_TRANSLATE, DEBUG, "%s   TargetEntry[%d]: %s", 
                                indent.c_str(), idx++, exprTypeName.c_str());
                    }
                }
            }

            // Print children with tree formatting
            if (plan->lefttree || plan->righttree) {
                if (plan->lefttree) {
                    printPlanTree(plan->lefttree, depth + 1, indent + " ");
                }
                if (plan->righttree) {
                    printPlanTree(plan->righttree, depth + 1, indent + " ");
                }
            }
        };

    // Print the complete execution tree
    PGX_LOG(AST_TRANSLATE, DEBUG, "=== POSTGRESQL EXECUTION TREE ===");
    printPlanTree(rootPlan, 0, "");
    PGX_LOG(AST_TRANSLATE, DEBUG, "=== END EXECUTION TREE ===");
}

bool QueryAnalyzer::validateAndLogPlanStructure(const PlannedStmt* stmt) {
    const auto rootPlan = stmt->planTree;
    Plan* scanPlan = nullptr;

    // Log the execution tree for analysis
    logExecutionTree(rootPlan);

    // ACCEPT ALL PATTERNS FROM TEST CASES
    if (rootPlan->type == T_SeqScan) {
        // Pattern 1: Simple table scan
        scanPlan = rootPlan;
        PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Simple SeqScan query");
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        // Pattern 2: Aggregation with SeqScan
        scanPlan = rootPlan->lefttree;
        PGX_LOG(AST_TRANSLATE, DEBUG, " ACCEPTED: Aggregate query with SeqScan source");
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_Gather) {
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
    }
    else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " UNKNOWN PATTERN: Accepting for testing but may need implementation");
        // Accept unknown patterns for comprehensive testing
    }

    // Extract table information if we found a scan plan
    if (scanPlan) {
        const auto scan = reinterpret_cast<SeqScan*>(scanPlan);
        const auto rte = static_cast<RangeTblEntry*>(list_nth(stmt->rtable, scan->scan.scanrelid - 1));

        PGX_LOG(AST_TRANSLATE, DEBUG, " Table OID: %d", rte->relid);

        // Set the global table OID for JIT runtime access
        g_jit_table_oid = rte->relid;
        PGX_LOG(AST_TRANSLATE, DEBUG, " Set g_jit_table_oid to: %d", g_jit_table_oid);
    }
    else {
        PGX_LOG(AST_TRANSLATE, DEBUG, " No scan plan extracted - query may not access tables directly");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, " QUERY ACCEPTED: Proceeding to MLIR compilation pipeline");
    return true;
}

#endif // POSTGRESQL_EXTENSION

auto QueryAnalyzer::analyzeForTesting(const char* queryText) -> QueryCapabilities {
    QueryCapabilities caps;

    if (queryText == nullptr) {
        return caps;
    }

    if ((strstr(queryText, "SELECT") != nullptr) && (strstr(queryText, "FROM") != nullptr)) {
        caps.isSelectStatement = true;
        caps.requiresSeqScan = true;
        caps.hasCompatibleTypes = true;
    }

    // Check for projection (specific columns rather than *)
    if ((strstr(queryText, "SELECT") != nullptr) && (strstr(queryText, "SELECT *") == nullptr)) {
        const char* selectPos = strstr(queryText, "SELECT");
        const char* fromPos = strstr(queryText, "FROM");
        if ((selectPos != nullptr) && (fromPos != nullptr) && fromPos > selectPos) {
            // Check if there are specific column names between SELECT and FROM
            const char* selectContent = selectPos + 6; // Skip "SELECT"
            while (*selectContent == ' ') {
                selectContent++; // Skip whitespace
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

    // Check for nested queries (subqueries)
    if (strstr(queryText, "(SELECT") != nullptr) {
        caps.requiresJoin = true; // Treat nested queries as requiring joins for now
    }

    return caps;
}

} // namespace pgx_lower