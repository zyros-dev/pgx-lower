#include "pgx-lower/execution/accepted_plan_verifier.h"

#include "pgx-lower/utility/logging.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include <stdexcept>
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

static auto mergeVerificationResult(AcceptedPlanVerificationResult& into, const AcceptedPlanVerificationResult& from)
    -> void {
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

auto AcceptedPlanVerificationResult::failure(AcceptedPlanVerificationPhase phase, std::string message,
                                             std::string location) -> AcceptedPlanVerificationResult {
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

auto AcceptedPlanVerificationResult::addFailure(AcceptedPlanVerificationPhase phase, std::string message,
                                                std::string location) -> void {
    failures_.push_back({phase, std::move(message), std::move(location)});
}

#ifdef POSTGRESQL_EXTENSION
static auto verifyTargetListMetadata(const List* targetList, const AcceptedPlanVerificationPhase phase,
                                     const std::string& location) -> AcceptedPlanVerificationResult {
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
    (void)stmt;
    (void)phase;
    return AcceptedPlanVerificationResult::success();
#endif
}

static auto verifyModuleEntryPoint(mlir::ModuleOp module, const AcceptedPlanVerificationPhase phase)
    -> AcceptedPlanVerificationResult {
    if (!module) {
        return AcceptedPlanVerificationResult::failure(phase, "MLIR module is null", "mlir.module");
    }

    if (phase == AcceptedPlanVerificationPhase::after_lowering) {
        if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main")
            || module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("_mlir_ciface_main"))
        {
            return AcceptedPlanVerificationResult::success();
        }
        return AcceptedPlanVerificationResult::failure(phase, "missing lowered main function", "mlir.module");
    }

    if (!module.lookupSymbol<mlir::func::FuncOp>("main")) {
        return AcceptedPlanVerificationResult::failure(phase, "missing main function", "mlir.module");
    }
    return AcceptedPlanVerificationResult::success();
}

static auto verifyNoHighLevelDialectsAfterLowering(mlir::ModuleOp module, const AcceptedPlanVerificationPhase phase)
    -> AcceptedPlanVerificationResult {
    auto result = AcceptedPlanVerificationResult::success();
    if (phase != AcceptedPlanVerificationPhase::after_lowering || !module) {
        return result;
    }

    module->walk([&](mlir::Operation* op) {
        const auto* dialect = op->getDialect();
        if (!dialect) {
            return;
        }

        const auto ns = dialect->getNamespace();
        if (ns == "relalg" || ns == "db" || ns == "dsa" || ns == "util") {
            result.addFailure(
                phase, "high-level dialect operation remains after lowering: " + op->getName().getStringRef().str(),
                "mlir.module");
        }
    });
    return result;
}

auto verifyAcceptedPlanModule(const PlannedStmt* stmt, mlir::ModuleOp module, const AcceptedPlanVerificationPhase phase)
    -> AcceptedPlanVerificationResult {
    auto result = verifyAcceptedPlanMetadata(stmt, phase);
    if (phase == AcceptedPlanVerificationPhase::after_lowering) {
        mergeVerificationResult(result, verifyNoHighLevelDialectsAfterLowering(module, phase));
        mergeVerificationResult(result, verifyModuleEntryPoint(module, phase));
    } else {
        mergeVerificationResult(result, verifyModuleEntryPoint(module, phase));
    }
    return result;
}

auto verifyAcceptedPlanOrThrow(const PlannedStmt* stmt, mlir::ModuleOp module, const AcceptedPlanVerificationPhase phase)
    -> void {
    const auto result = verifyAcceptedPlanModule(stmt, module, phase);
    if (result.ok()) {
        return;
    }

    PGX_ERROR("Accepted plan verifier failed: %s", result.summary().c_str());
    throw std::runtime_error("Accepted plan verifier failed: " + result.summary());
}

} // namespace pgx_lower::execution
