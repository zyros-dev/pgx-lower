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

enum class LowerPath {
    not_applicable,
    row,
    legacy,
};

auto lowerPathName(LowerPath path) -> const char*;

struct UnsupportedReason {
    UnsupportedReasonKind kind = UnsupportedReasonKind::invalid;
    std::string message;
    std::string location;
};

class AnalyzerResult {
   public:
    AnalyzerResult();

    static auto supported() -> AnalyzerResult;
    static auto unsupported(UnsupportedReasonKind kind, std::string message, std::string location = {}) -> AnalyzerResult;

    [[nodiscard]] auto isSupported() const -> bool;
    [[nodiscard]] auto reasons() const -> const std::vector<UnsupportedReason>&;
    [[nodiscard]] auto primaryReason() const -> const UnsupportedReason&;
    [[nodiscard]] auto primaryReasonKindName() const -> std::string;
    [[nodiscard]] auto humanSummary() const -> std::string;

    auto addUnsupportedReason(UnsupportedReasonKind kind, std::string message, std::string location = {}) -> void;

   private:
    bool supported_ = false;
    std::vector<UnsupportedReason> reasons_;
};

class QueryAnalyzer {
   public:
#ifdef POSTGRESQL_EXTENSION
    static AnalyzerResult analyzePlan(const PlannedStmt* stmt);
    static AnalyzerResult analyzeNode(const Plan* plan, std::string location = "Plan");
    static AnalyzerResult analyzeExpr(const Node* expr, const std::string& location);
    static AnalyzerResult analyzeNodeForTesting(const Plan* plan);
    static AnalyzerResult analyzeExprForTesting(const Node* expr);
    static LowerPath classifyLowerPath(const PlannedStmt* stmt);
    static LowerPath classifyLowerPathForTesting(const Plan* plan);

    static void logExecutionTree(Plan* rootPlan);
    static bool validateAndLogPlanStructure(const PlannedStmt* stmt);
#endif

   private:
#ifdef POSTGRESQL_EXTENSION
    static AnalyzerResult analyzeTargetList(const List* targetList, const std::string& location);
    static AnalyzerResult analyzeExprList(const List* expressions, const std::string& location);
    static AnalyzerResult analyzePlanTargetTypes(const Plan* plan, std::string location);
    static AnalyzerResult analyzeExprType(const Node* expr, std::string location);
    static bool checkCommandType(const PlannedStmt* stmt);
    static bool isTypeSupportedByMLIR(Oid postgresType);
    static bool isFunctionSupported(const FuncExpr* func);
    static bool isAggregateSupported(const Aggref* agg);
    static bool isOperatorSupported(const OpExpr* op);
    static bool isCollationSupported(Oid collationOid);
#endif
};

} // namespace pgx_lower
