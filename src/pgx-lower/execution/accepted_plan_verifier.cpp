#include "pgx-lower/execution/accepted_plan_verifier.h"

#include <utility>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/nodeFuncs.h"
}
#endif

namespace pgx_lower::execution {

static auto mergeVerificationResult(AcceptedPlanVerificationResult& into,
                                    const AcceptedPlanVerificationResult& from) -> void {
    for (const auto& failure : from.failures()) {
        into.addFailure(failure.phase, failure.message, failure.location);
    }
}

auto acceptedPlanVerificationPhaseName(const AcceptedPlanVerificationPhase phase) -> const char* {
    switch (phase) {
    case AcceptedPlanVerificationPhase::after_ast_translation: return "after_ast_translation";
    case AcceptedPlanVerificationPhase::after_lowering: return "after_lowering";
    }
    return "after_ast_translation";
}

auto AcceptedPlanVerificationResult::success() -> AcceptedPlanVerificationResult {
    return {};
}

auto AcceptedPlanVerificationResult::failure(AcceptedPlanVerificationPhase phase,
                                             std::string message,
                                             std::string location)
    -> AcceptedPlanVerificationResult {
    AcceptedPlanVerificationResult result;
    result.addFailure(phase, std::move(message), std::move(location));
    return result;
}

auto AcceptedPlanVerificationResult::ok() const -> bool {
    return failures_.empty();
}

auto AcceptedPlanVerificationResult::failures() const -> const std::vector<AcceptedPlanVerificationFailure>& {
    return failures_;
}

auto AcceptedPlanVerificationResult::summary() const -> std::string {
    if (ok()) {
        return "accepted plan verifier passed";
    }

    const auto& failure = failures_.front();
    auto text = std::string(acceptedPlanVerificationPhaseName(failure.phase)) + ": " + failure.message;
    if (!failure.location.empty()) {
        text += " at " + failure.location;
    }
    return text;
}

auto AcceptedPlanVerificationResult::addFailure(AcceptedPlanVerificationPhase phase,
                                                std::string message,
                                                std::string location) -> void {
    failures_.push_back({phase, std::move(message), std::move(location)});
}

#ifdef POSTGRESQL_EXTENSION
static auto verifyTargetListMetadata(const List* targetList,
                                     const AcceptedPlanVerificationPhase phase,
                                     const std::string& location)
    -> AcceptedPlanVerificationResult {
    auto result = AcceptedPlanVerificationResult::success();
    if (!targetList) {
        result.addFailure(phase, "target list is null", location);
        return result;
    }

    ListCell* lc = nullptr;
    auto index = 0;
    foreach (lc, targetList) {
        const auto* tle = static_cast<const TargetEntry*>(lfirst(lc));
        if (!tle || tle->resjunk) {
            ++index;
            continue;
        }

        const auto itemLocation = location + "[" + std::to_string(index) + "]";
        if (!tle->expr) {
            result.addFailure(phase, "target expression is null", itemLocation + ".expr");
            ++index;
            continue;
        }

        const auto typeOid = exprType(const_cast<Node*>(reinterpret_cast<const Node*>(tle->expr)));
        if (typeOid == InvalidOid) {
            result.addFailure(phase, "target expression type OID is invalid", itemLocation + ".type");
        }
        ++index;
    }
    return result;
}
#endif

auto verifyAcceptedPlanMetadata(const PlannedStmt* stmt, const AcceptedPlanVerificationPhase phase)
    -> AcceptedPlanVerificationResult {
#ifdef POSTGRESQL_EXTENSION
    if (!stmt) {
        return AcceptedPlanVerificationResult::failure(phase, "planned statement is null", "PlannedStmt");
    }
    if (!stmt->planTree) {
        return AcceptedPlanVerificationResult::failure(phase, "plan tree is null", "PlannedStmt.planTree");
    }

    auto result = AcceptedPlanVerificationResult::success();
    mergeVerificationResult(result, verifyTargetListMetadata(stmt->planTree->targetlist, phase, "Plan.targetlist"));
    return result;
#else
    (void) stmt;
    (void) phase;
    return AcceptedPlanVerificationResult::success();
#endif
}

} // namespace pgx_lower::execution
