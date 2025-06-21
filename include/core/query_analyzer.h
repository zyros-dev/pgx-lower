#pragma once

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
}
#endif

namespace pgx_lower {

/**
 * Describes what MLIR operations a query requires
 */
struct QueryCapabilities {
    bool requiresSeqScan = false;        // Sequential table scan
    bool requiresFilter = false;         // WHERE clause filtering  
    bool requiresProjection = false;     // Column selection
    bool requiresAggregation = false;    // SUM, COUNT, etc.
    bool requiresJoin = false;           // Table joins
    bool requiresSort = false;           // ORDER BY
    bool requiresLimit = false;          // LIMIT clause
    
    // Check if MLIR can handle this combination
    bool isMLIRCompatible() const;
    
    // Get human-readable description
    const char* getDescription() const;
};

/**
 * Analyzes PostgreSQL query plans to determine MLIR compatibility
 */
class QueryAnalyzer {
public:
    // Analyze a PostgreSQL plan node
    #ifdef POSTGRESQL_EXTENSION
    static QueryCapabilities analyzePlan(const PlannedStmt* stmt);
    static QueryCapabilities analyzeNode(const Plan* plan);
    #endif
    
    // Mock analysis for unit tests
    static QueryCapabilities analyzeForTesting(const char* queryText);
    
private:
    #ifdef POSTGRESQL_EXTENSION
    static void analyzeSeqScan(const SeqScan* seqScan, QueryCapabilities& caps);
    static void analyzeFilter(const Plan* plan, QueryCapabilities& caps);
    static void analyzeProjection(const Plan* plan, QueryCapabilities& caps);
    #endif
};

} // namespace pgx_lower