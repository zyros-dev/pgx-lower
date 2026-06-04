#pragma once
#include <utility>
#include <vector>
#include <string>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "nodes/primnodes.h"
}
#endif


namespace pgx_lower {

enum class UnsupportedReasonKind {
    invalid,
    unsupported_plan_node,
    unsupported_expr_node,
    unsupported_type,
    unsupported_operator,
    unsupported_function,
    unsupported_collation,
    missing_metadata,
};

auto unsupportedReasonKindName(UnsupportedReasonKind kind) -> const char*;

struct UnsupportedReason {
    UnsupportedReasonKind kind = UnsupportedReasonKind::invalid;
    std::string message;
    std::string location;
};

class AnalyzerResult {
   public:
    AnalyzerResult();

    static auto supported() -> AnalyzerResult;
    static auto unsupported(UnsupportedReasonKind kind,
                            std::string message,
                            std::string location = {}) -> AnalyzerResult;

    [[nodiscard]] auto isSupported() const -> bool;
    [[nodiscard]] auto reasons() const -> const std::vector<UnsupportedReason>&;
    [[nodiscard]] auto primaryReason() const -> const UnsupportedReason&;
    [[nodiscard]] auto primaryReasonKindName() const -> std::string;
    [[nodiscard]] auto humanSummary() const -> std::string;

    auto addUnsupportedReason(UnsupportedReasonKind kind,
                              std::string message,
                              std::string location = {}) -> void;

   private:
    bool supported_ = false;
    std::vector<UnsupportedReason> reasons_;
};

struct QueryCapabilities {
    bool requiresSeqScan = false;
    bool requiresFilter = false;
    bool requiresProjection = false;
    bool requiresAggregation = false;
    bool requiresJoin = false;
    bool requiresSort = false;
    bool requiresLimit = false;
    bool isSelectStatement = false;
    bool hasCompatibleTypes = false;
    bool hasExpressions = false;

    [[nodiscard]] auto isMLIRCompatible() const -> bool;

    [[nodiscard]] auto getDescription() const -> std::string;
};

class QueryAnalyzer {
   public:
#ifdef POSTGRESQL_EXTENSION
    static QueryCapabilities analyzePlan(const PlannedStmt* stmt);
    static QueryCapabilities analyzeNode(const Plan* plan);
    
    static void logExecutionTree(Plan* rootPlan);
    static bool validateAndLogPlanStructure(const PlannedStmt* stmt);
#endif

    static auto analyzeForTesting(const char* queryText) -> QueryCapabilities;

   private:
#ifdef POSTGRESQL_EXTENSION
    static void analyzeSeqScan(const SeqScan* seqScan, QueryCapabilities& caps);
    static void analyzeFilter(const Plan* plan, QueryCapabilities& caps);
    static void analyzeProjection(const Plan* plan, QueryCapabilities& caps);
    static void analyzeTypes(const Plan* plan, QueryCapabilities& caps);
    static bool checkCommandType(const PlannedStmt* stmt);
    static bool isTypeSupportedByMLIR(Oid postgresType);
    static std::pair<int, int> analyzeTypeCompatibility(const std::vector<Oid>& types);
#endif
};

} // namespace pgx_lower
