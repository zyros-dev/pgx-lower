#include "pgx-lower/execution/accepted_plan_verifier.h"

#include <utility>

namespace pgx_lower::execution {

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

} // namespace pgx_lower::execution
