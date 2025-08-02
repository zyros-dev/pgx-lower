#include "core/query_analyzer.h"
#include "core/mlir_logger.h" 
#include "core/error_handling.h"
#include "core/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "nodes/primnodes.h"
#include "nodes/nodeFuncs.h"
#include "utils/lsyscache.h"
}
#include "postgres/executor_c.h"
#endif

#include <cstring>
#include <vector>

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
    
    // Expressions should work through MLIR - no fallback
    // TODO: Fix MLIR expression compilation instead of avoiding it
    
    return compatible;
}

auto QueryCapabilities::getDescription() const -> const char* {
    if (isMLIRCompatible()) {
        return "Sequential scan with optional aggregation - MLIR compatible";
    }

    static thread_local char description[256];
    char* pos = description;

    strcpy(pos, "Requires: ");
    pos += strlen(pos);

    bool needComma = false;

    if (requiresSeqScan) {
        strcpy(pos, "SeqScan");
        pos += strlen(pos);
        needComma = true;
    }

    if (requiresFilter) {
        if (needComma) {
            strcpy(pos, ", ");
            pos += 2;
        }
        strcpy(pos, "Filter");
        pos += strlen(pos);
        needComma = true;
    }

    if (requiresProjection) {
        if (needComma) {
            strcpy(pos, ", ");
            pos += 2;
        }
        strcpy(pos, "Projection");
        pos += strlen(pos);
        needComma = true;
    }

    if (requiresAggregation) {
        if (needComma) {
            strcpy(pos, ", ");
            pos += 2;
        }
        strcpy(pos, "Aggregation");
        pos += strlen(pos);
        needComma = true;
    }

    if (requiresJoin) {
        if (needComma) {
            strcpy(pos, ", ");
            pos += 2;
        }
        strcpy(pos, "Join");
        pos += strlen(pos);
        needComma = true;
    }

    if (requiresSort) {
        if (needComma) {
            strcpy(pos, ", ");
            pos += 2;
        }
        strcpy(pos, "Sort");
        pos += strlen(pos);
        needComma = true;
    }

    if (requiresLimit) {
        if (needComma) {
            strcpy(pos, ", ");
            pos += 2;
        }
        strcpy(pos, "Limit");
        pos += strlen(pos);
        needComma = true;
    }

    if (hasExpressions) {
#ifdef POSTGRESQL_EXTENSION
        if (::g_extension_after_load) {
            if (needComma) {
                strcpy(pos, ", ");
                pos += 2;
            }
            strcpy(pos, "Expressions (disabled after LOAD)");
            pos += strlen(pos);
            needComma = true;
        }
#endif
    }

    strcpy(pos, " - Not yet supported by MLIR");

    return description;
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
        TargetEntry* tle = (TargetEntry*)lfirst(lc);
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
            
            Oid columnType = exprType((Node*)tle->expr);
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