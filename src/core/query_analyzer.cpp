#include "core/query_analyzer.h"
#include "core/mlir_logger.h"
#include "core/error_handling.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "utils/lsyscache.h"
}
#endif

#include <cstring>

namespace pgx_lower {

bool QueryCapabilities::isMLIRCompatible() const {
    // Currently, MLIR supports simple sequential scans and column projection
    // No filters, aggregations, joins, sorts, or limits yet
    return requiresSeqScan && !requiresFilter && !requiresAggregation && !requiresJoin && !requiresSort && !requiresLimit;
    // Note: projection is now supported, so requiresProjection is allowed
}

const char* QueryCapabilities::getDescription() const {
    if (isMLIRCompatible()) {
        return "Simple sequential scan - MLIR compatible";
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
    }

    strcpy(pos, " - Not yet supported by MLIR");

    return description;
}

#ifdef POSTGRESQL_EXTENSION

QueryCapabilities QueryAnalyzer::analyzePlan(const PlannedStmt* stmt) {
    if (!stmt || !stmt->planTree) {
        auto error = ErrorManager::queryAnalysisError("No plan tree to analyze");
        ErrorManager::reportError(error);
        return {};
    }

    elog(DEBUG1, "Analyzing PostgreSQL plan for MLIR compatibility");

    try {
        return analyzeNode(stmt->planTree);
    } catch (const std::exception& e) {
        auto error = ErrorManager::queryAnalysisError("Exception during plan analysis: " + std::string(e.what()));
        ErrorManager::reportError(error);
        return {};
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
        elog(DEBUG1, "Index scans not yet supported by MLIR");
        caps.requiresSeqScan = false; // This is an index scan, not seq scan
        break;

    case T_NestLoop:
    case T_MergeJoin:
    case T_HashJoin:
        elog(DEBUG1, "Join operations not yet supported by MLIR");
        caps.requiresJoin = true;
        break;

    case T_Sort:
        elog(DEBUG1, "Sort operations not yet supported by MLIR");
        caps.requiresSort = true;
        break;

    case T_Limit:
        elog(DEBUG1, "Limit operations not yet supported by MLIR");
        caps.requiresLimit = true;
        break;

    case T_Agg:
        elog(DEBUG1, "Aggregation operations not yet supported by MLIR");
        caps.requiresAggregation = true;
        break;

    default: elog(DEBUG1, "Unknown plan node type: %d", nodeTag(plan)); break;
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
    elog(DEBUG1, "Found sequential scan on table");
    caps.requiresSeqScan = true;
}

void QueryAnalyzer::analyzeFilter(const Plan* plan, QueryCapabilities& caps) {
    if (plan->qual) {
        elog(DEBUG1, "Found WHERE clause - filtering required");
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
        elog(DEBUG1, "Target list found - basic projection handling available");
    }
}

#endif // POSTGRESQL_EXTENSION

QueryCapabilities QueryAnalyzer::analyzeForTesting(const char* queryText) {
    QueryCapabilities caps;

    if (!queryText) {
        return caps;
    }

    // Simple string-based analysis for unit tests
    if (strstr(queryText, "SELECT") && strstr(queryText, "FROM")) {
        caps.requiresSeqScan = true;
    }

    // Check for projection (specific columns rather than *)
    if (strstr(queryText, "SELECT") && !strstr(queryText, "SELECT *")) {
        const char* selectPos = strstr(queryText, "SELECT");
        const char* fromPos = strstr(queryText, "FROM");
        if (selectPos && fromPos && fromPos > selectPos) {
            // Check if there are specific column names between SELECT and FROM
            const char* selectContent = selectPos + 6; // Skip "SELECT"
            while (*selectContent == ' ')
                selectContent++; // Skip whitespace
            if (selectContent < fromPos && *selectContent != '*') {
                caps.requiresProjection = true;
            }
        }
    }

    if (strstr(queryText, "WHERE")) {
        caps.requiresFilter = true;
    }

    if (strstr(queryText, "JOIN")) {
        caps.requiresJoin = true;
    }

    if (strstr(queryText, "ORDER BY")) {
        caps.requiresSort = true;
    }

    if (strstr(queryText, "LIMIT")) {
        caps.requiresLimit = true;
    }

    if (strstr(queryText, "COUNT") || strstr(queryText, "SUM") || strstr(queryText, "AVG")
        || strstr(queryText, "GROUP BY"))
    {
        caps.requiresAggregation = true;
    }

    // Check for nested queries (subqueries)
    if (strstr(queryText, "(SELECT")) {
        caps.requiresJoin = true; // Treat nested queries as requiring joins for now
    }

    return caps;
}

} // namespace pgx_lower