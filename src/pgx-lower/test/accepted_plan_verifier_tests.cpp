extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "nodes/plannodes.h"
}

#include "pgx-lower/execution/accepted_plan_verifier.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

PGX_TEST_FN(accepted_plan_verifier_success_has_no_failures) {
    const auto result = pgx_lower::execution::AcceptedPlanVerificationResult::success();
    REQUIRE(result.ok());
    REQUIRE(result.failures().empty());
    REQUIRE(result.summary() == std::string("accepted plan verifier passed"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(accepted_plan_verifier_failure_summary_names_phase) {
    auto result = pgx_lower::execution::AcceptedPlanVerificationResult::failure(
        pgx_lower::execution::AcceptedPlanVerificationPhase::after_ast_translation,
        "missing main function",
        "mlir.module");
    REQUIRE(!result.ok());
    REQUIRE(result.failures().size() == 1);
    REQUIRE(result.summary() == std::string("after_ast_translation: missing main function at mlir.module"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(accepted_plan_verifier_rejects_null_planned_stmt) {
    const auto result = pgx_lower::execution::verifyAcceptedPlanMetadata(
        nullptr,
        pgx_lower::execution::AcceptedPlanVerificationPhase::after_ast_translation);
    REQUIRE(!result.ok());
    REQUIRE(result.summary() == std::string("after_ast_translation: planned statement is null at PlannedStmt"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(accepted_plan_verifier_rejects_missing_plan_tree) {
    auto stmt = PlannedStmt{};
    stmt.planTree = nullptr;
    const auto result = pgx_lower::execution::verifyAcceptedPlanMetadata(
        &stmt,
        pgx_lower::execution::AcceptedPlanVerificationPhase::after_ast_translation);
    REQUIRE(!result.ok());
    REQUIRE(result.summary() == std::string("after_ast_translation: plan tree is null at PlannedStmt.planTree"));
    PG_RETURN_VOID();
}
