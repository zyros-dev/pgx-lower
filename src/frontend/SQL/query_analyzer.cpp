#include "frontend/SQL/query_analyzer.h"
 
#include "execution/error_handling.h"
#include "execution/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "nodes/primnodes.h"
#include "nodes/nodeFuncs.h"
#include "utils/lsyscache.h"

// Forward declaration for table OID global variable
extern Oid g_jit_table_oid;
}
#include "execution/postgres/executor_c.h"
#endif

#include <cstring>
#include <vector>
#include <sstream>
#include <functional>

#ifdef POSTGRESQL_EXTENSION
// Forward declaration of global flag defined in executor_c.cpp
extern bool g_extension_after_load;
#endif

namespace pgx_lower {

// NOTE: I break C++ rules a bit in this file since it's interacting a lot with C-style things.

auto QueryCapabilities::isMLIRCompatible() const -> bool {
    // All conditions must be met for MLIR compatibility
    bool compatible = isSelectStatement &&           // Only SELECT statements
                      hasCompatibleTypes &&          // All types must be MLIR-supported
                      requiresSeqScan &&              // Must have sequential scan
                      !requiresJoin &&                // No joins yet
                      !requiresSort &&                // No sorting yet  
                      !requiresLimit &&               // No limits yet
                      !requiresFilter;                // No WHERE clauses yet (temporary)
    // Note: requiresAggregation is allowed and supported (SUM, COUNT, etc.)
    // Note: requiresFilter temporarily disabled for debugging

    return compatible;
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

    PGX_DEBUG("Analyzing PostgreSQL plan for MLIR compatibility");

    try {
        // 1. Check command type first (CMD_SELECT only)
        caps.isSelectStatement = checkCommandType(stmt);
        if (!caps.isSelectStatement) {
            PGX_DEBUG("Not a SELECT statement, MLIR not compatible");
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
        PGX_DEBUG("Index scans not yet supported by MLIR");
        caps.requiresSeqScan = false; // This is an index scan, not seq scan
        break;

    case T_NestLoop:
    case T_MergeJoin:
    case T_HashJoin:
        PGX_DEBUG("Join operations not yet supported by MLIR");
        caps.requiresJoin = true;
        break;

    case T_Sort:
        PGX_DEBUG("Sort operations not yet supported by MLIR");
        caps.requiresSort = true;
        break;

    case T_Limit:
        PGX_DEBUG("Limit operations not yet supported by MLIR");
        caps.requiresLimit = true;
        break;

    case T_Agg:
        PGX_DEBUG("Aggregation operations detected - MLIR support available");
        caps.requiresAggregation = true;
        break;

    default: PGX_DEBUG("Unknown plan node type: " + std::to_string(nodeTag(plan))); break;
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
    PGX_DEBUG("Found sequential scan on table");
    caps.requiresSeqScan = true;
}

void QueryAnalyzer::analyzeFilter(const Plan* plan, QueryCapabilities& caps) {
    if (plan->qual) {
        PGX_DEBUG("Found WHERE clause - filtering required");
        caps.requiresFilter = true;
    }
}

void QueryAnalyzer::analyzeProjection(const Plan* plan, QueryCapabilities& caps) {
    // For now, assume any non-trivial target list requires projection
    // In the future, we can be more sophisticated about this
    if (plan->targetlist) {
        // Simple heuristic: if we have a target list, we might need projection
        // For now, we'll be conservative and not mark this as requiring projection
        // since our current MLIR implementation handles basic projections
        PGX_DEBUG("Target list found - basic projection handling available");
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
    foreach(lc, plan->targetlist) {
        TargetEntry* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle && !tle->resjunk && tle->expr) {
            // Check if this is a computed expression (not just a simple Var)
            if (nodeTag(tle->expr) != T_Var) {
                caps.hasExpressions = true;
                PGX_DEBUG("Found computed expression in target list (node type: " + std::to_string(nodeTag(tle->expr)) + ")");
            }
            
            // For now, allow arithmetic expressions to test the new RelAlg Map implementation
            // Later we can add more sophisticated filtering
            if (IsA(tle->expr, FuncExpr)) {
                PGX_DEBUG("Found function expression in target list - marking as incompatible");
                caps.hasCompatibleTypes = false;
                return;
            }
            
            Oid columnType = exprType(reinterpret_cast<Node*>(tle->expr));
            columnTypes.push_back(columnType);
        }
    }

    if (columnTypes.empty()) {
        PGX_DEBUG("No columns found in target list");
        caps.hasCompatibleTypes = false;
        return;
    }

    // Analyze type compatibility using built-in system
    auto [supportedCount, unsupportedCount] = analyzeTypeCompatibility(columnTypes);
    
    PGX_DEBUG("Type analysis: " + std::to_string(supportedCount) + " supported, " + std::to_string(unsupportedCount) + " unsupported out of " + std::to_string(columnTypes.size()) + " total columns");

    // For now, require ALL types to be supported
    // In the future, we could allow partial support with fallbacks
    caps.hasCompatibleTypes = (unsupportedCount == 0);
    
    if (!caps.hasCompatibleTypes) {
        PGX_DEBUG("Some column types not supported by MLIR runtime");
    }
}

bool QueryAnalyzer::checkCommandType(const PlannedStmt* stmt) {
    if (!stmt) {
        return false;
    }
    
    bool isSelect = (stmt->commandType == CMD_SELECT);
    if (!isSelect) {
        PGX_DEBUG("Command type " + std::to_string(stmt->commandType) + " is not SELECT, MLIR only supports SELECT statements");
    }
    
    return isSelect;
}

bool QueryAnalyzer::isTypeSupportedByMLIR(Oid postgresType) {
    // PostgreSQL types that MLIR runtime can handle
    // Based on working test cases and available runtime functions
    switch (postgresType) {
        // Integer types (handled by get_int_field)
        case BOOLOID:
        case INT2OID:
        case INT4OID:
        case INT8OID:
        case NUMERICOID:
        case FLOAT4OID:
        case FLOAT8OID:
            return true;
            
        case TEXTOID:
        case VARCHAROID:
        case CHAROID:
            return true;

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
        case PG_LSNOID:
            return true;

        default:
            return false;
    }
}

std::pair<int, int> QueryAnalyzer::analyzeTypeCompatibility(const std::vector<Oid>& types) {
    int supportedCount = 0;
    int unsupportedCount = 0;
    
    for (Oid type : types) {
        if (isTypeSupportedByMLIR(type)) {
            supportedCount++;
        } else {
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
                PGX_INFO(prefix + "NULL");
                return;
            }
            
            std::string indent(depth * 2, ' ');
            std::string nodeName = getNodeTypeName(static_cast<NodeTag>(plan->type));
            std::string nodeInfo = prefix + nodeName + " (type=" + std::to_string(plan->type) + ")";
            
            // Add node-specific details
            if (plan->type == T_SeqScan) {
                auto* seqScan = reinterpret_cast<SeqScan*>(plan);
                nodeInfo += " [scanrelid=" + std::to_string(seqScan->scan.scanrelid) + "]";
            } else if (plan->type == T_Agg) {
                auto* agg = reinterpret_cast<Agg*>(plan);
                nodeInfo += " [strategy=" + std::to_string(agg->aggstrategy) + "]";
            } else if (plan->type == T_Gather) {
                auto* gather = reinterpret_cast<Gather*>(plan);
                nodeInfo += " [num_workers=" + std::to_string(gather->num_workers) + "]";
            }
            
            PGX_INFO(nodeInfo);
            
            // Print children with tree formatting
            if (plan->lefttree || plan->righttree) {
                if (plan->lefttree) {
                    printPlanTree(plan->lefttree, depth + 1, indent + "├─ ");
                }
                if (plan->righttree) {
                    printPlanTree(plan->righttree, depth + 1, indent + "└─ ");
                }
            }
        };
    
    // Print the complete execution tree
    PGX_INFO("=== POSTGRESQL EXECUTION TREE ===");
    printPlanTree(rootPlan, 0, "");
    PGX_INFO("=== END EXECUTION TREE ===");
}

bool QueryAnalyzer::validateAndLogPlanStructure(const PlannedStmt* stmt) {
    const auto rootPlan = stmt->planTree;
    Plan* scanPlan = nullptr;

    logExecutionTree(rootPlan);
    if (rootPlan->type == T_SeqScan) {
        scanPlan = rootPlan;
        PGX_DEBUG("Detected simple SeqScan query");
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        scanPlan = rootPlan->lefttree;
        PGX_DEBUG("Detected aggregate query with SeqScan source");
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_Gather) {
        // Handle parallel aggregation: Agg → Gather → Agg → SeqScan
        auto* gatherPlan = rootPlan->lefttree;
        if (gatherPlan->lefttree && gatherPlan->lefttree->type == T_Agg) {
            auto* innerAggPlan = gatherPlan->lefttree;
            if (innerAggPlan->lefttree && innerAggPlan->lefttree->type == T_SeqScan) {
                scanPlan = innerAggPlan->lefttree;
                PGX_DEBUG("Detected parallel aggregate query (Agg→Gather→Agg→SeqScan)");
            }
        }
        
        if (!scanPlan) {
            PGX_ERROR("Query analyzer: Parallel aggregation pattern not fully supported yet");
            return false;
        }
    }
    else {
        PGX_ERROR("Query analyzer: Unsupported execution pattern - need to extend compatibility logic");
        return false;
    }

    const auto scan = reinterpret_cast<SeqScan*>(scanPlan);
    const auto rte = static_cast<RangeTblEntry*>(list_nth(stmt->rtable, scan->scan.scanrelid - 1));

    PGX_DEBUG("Using AST-based translation - JIT will manage table scan");
    PGX_INFO("Table OID: " + std::to_string(rte->relid));
    
    // Set the global table OID for JIT runtime access
    g_jit_table_oid = rte->relid;
    PGX_INFO("Set g_jit_table_oid to: " + std::to_string(g_jit_table_oid));

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

    if ((strstr(queryText, "COUNT") != nullptr) || (strstr(queryText, "SUM") != nullptr) || (strstr(queryText, "AVG") != nullptr)
        || (strstr(queryText, "GROUP BY") != nullptr))
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