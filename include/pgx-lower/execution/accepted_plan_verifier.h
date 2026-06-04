#pragma once

#include <string>
#include <vector>

extern "C" {
struct PlannedStmt;
}

namespace mlir {
class ModuleOp;
}

namespace pgx_lower::execution {

enum class AcceptedPlanVerificationPhase {
    after_ast_translation,
    after_lowering,
};

auto acceptedPlanVerificationPhaseName(AcceptedPlanVerificationPhase phase) -> const char*;

struct AcceptedPlanVerificationFailure {
    AcceptedPlanVerificationPhase phase = AcceptedPlanVerificationPhase::after_ast_translation;
    std::string message;
    std::string location;
};

class AcceptedPlanVerificationResult {
   public:
    static auto success() -> AcceptedPlanVerificationResult;
    static auto failure(AcceptedPlanVerificationPhase phase,
                        std::string message,
                        std::string location = {}) -> AcceptedPlanVerificationResult;

    [[nodiscard]] auto ok() const -> bool;
    [[nodiscard]] auto failures() const -> const std::vector<AcceptedPlanVerificationFailure>&;
    [[nodiscard]] auto summary() const -> std::string;

    auto addFailure(AcceptedPlanVerificationPhase phase, std::string message, std::string location = {}) -> void;

   private:
    std::vector<AcceptedPlanVerificationFailure> failures_;
};

auto verifyAcceptedPlanMetadata(const PlannedStmt* stmt, AcceptedPlanVerificationPhase phase)
    -> AcceptedPlanVerificationResult;
auto verifyAcceptedPlanModule(const PlannedStmt* stmt,
                              mlir::ModuleOp module,
                              AcceptedPlanVerificationPhase phase)
    -> AcceptedPlanVerificationResult;
auto verifyAcceptedPlanOrThrow(const PlannedStmt* stmt,
                               mlir::ModuleOp module,
                               AcceptedPlanVerificationPhase phase) -> void;

} // namespace pgx_lower::execution
