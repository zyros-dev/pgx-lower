#pragma once

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
}
#endif

namespace pgx_lower {

struct QueryCapabilities {
    bool requiresSeqScan = false; // Sequential table scan
    bool requiresFilter = false; // WHERE clause filtering
    bool requiresProjection = false; // Column selection
    bool requiresAggregation = false; // SUM, COUNT, etc.
    bool requiresJoin = false; // Table joins
    bool requiresSort = false; // ORDER BY
    bool requiresLimit = false; // LIMIT clause
    bool isSelectStatement = false; // Only SELECT statements supported
    bool hasCompatibleTypes = false; // All column types are MLIR-compatible

    // Check if MLIR can handle this combination
    [[nodiscard]] auto isMLIRCompatible() const -> bool;

    // Get human-readable description
    [[nodiscard]] auto getDescription() const -> const char*;
};

class QueryAnalyzer {
   public:
// Analyze a PostgreSQL plan node
#ifdef POSTGRESQL_EXTENSION
    static QueryCapabilities analyzePlan(const PlannedStmt* stmt);
    static QueryCapabilities analyzeNode(const Plan* plan);
#endif

    // Mock analysis for unit tests
    static auto analyzeForTesting(const char* queryText) -> QueryCapabilities;

   private:
#ifdef POSTGRESQL_EXTENSION
    static void analyzeSeqScan(const SeqScan* seqScan, QueryCapabilities& caps);
    static void analyzeFilter(const Plan* plan, QueryCapabilities& caps);
    static void analyzeProjection(const Plan* plan, QueryCapabilities& caps);
#endif
};

} // namespace pgx_lower