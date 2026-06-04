extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/pg_list.h"
}

#include "pgx-lower/execution/accepted_plan_verifier.h"
#include "pgx-lower/test/pgx_test_fn.h"
#include "pgx-lower/test/standalone_mlir_runner.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <stdexcept>
#include <string>

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

namespace {

struct ValidPlanFixture {
    Const value{};
    TargetEntry target{};
    Plan plan{};
    PlannedStmt stmt{};

    ValidPlanFixture() {
        value.xpr.type = T_Const;
        value.consttype = INT4OID;
        value.consttypmod = -1;
        value.constvalue = Datum{1};
        value.constisnull = false;
        value.constbyval = true;
        value.constlen = sizeof(int32);

        target.xpr.type = T_TargetEntry;
        target.expr = reinterpret_cast<Expr*>(&value);
        target.resjunk = false;

        plan.type = T_SeqScan;
        plan.targetlist = list_make1(&target);
        stmt.planTree = &plan;
    }
};

} // namespace

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

PGX_TEST_FN(accepted_plan_verifier_rejects_missing_main_function) {
    ValidPlanFixture fixture;
    mlir::MLIRContext context;
    const auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    const auto result = pgx_lower::execution::verifyAcceptedPlanModule(
        &fixture.stmt,
        module,
        pgx_lower::execution::AcceptedPlanVerificationPhase::after_ast_translation);
    REQUIRE(!result.ok());
    REQUIRE(result.summary() == std::string("after_ast_translation: missing main function at mlir.module"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(accepted_plan_verifier_rejects_high_level_op_after_lowering) {
    ValidPlanFixture fixture;
    auto tester = std::make_unique<pgx_test::StandalonePipelineTester>();

    const char* relAlgMLIR = R"(
        module {
          func.func @main() -> !dsa.table {
            %0 = relalg.basetable  {column_order = ["id"], table_identifier = "test|oid:32970940"} columns: {id => @test::@id({type = i32})}
            %1 = relalg.materialize %0 [@test::@id] => ["id"] : !dsa.table
            return %1 : !dsa.table
          }
        }
    )";
    REQUIRE(tester->loadRelAlgModule(relAlgMLIR));

    const auto result = pgx_lower::execution::verifyAcceptedPlanModule(
        &fixture.stmt,
        tester->getModule(),
        pgx_lower::execution::AcceptedPlanVerificationPhase::after_lowering);
    REQUIRE(!result.ok());
    REQUIRE(result.summary().find("after_lowering: high-level dialect operation remains after lowering:") == 0);
    PG_RETURN_VOID();
}

PGX_TEST_FN(accepted_plan_verifier_accepts_llvm_main_after_lowering) {
    ValidPlanFixture fixture;
    mlir::MLIRContext context;
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    const auto loc = mlir::UnknownLoc::get(&context);
    auto module = mlir::ModuleOp::create(loc);
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(module.getBody());
    const auto functionType = mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(&context), {});
    builder.create<mlir::LLVM::LLVMFuncOp>(loc, "main", functionType);

    const auto result = pgx_lower::execution::verifyAcceptedPlanModule(
        &fixture.stmt,
        module,
        pgx_lower::execution::AcceptedPlanVerificationPhase::after_lowering);
    REQUIRE(result.ok());
    PG_RETURN_VOID();
}

PGX_TEST_FN(accepted_plan_verifier_throw_names_internal_verifier) {
    auto stmt = PlannedStmt{};
    try {
        pgx_lower::execution::verifyAcceptedPlanOrThrow(
            &stmt,
            mlir::ModuleOp{},
            pgx_lower::execution::AcceptedPlanVerificationPhase::after_lowering);
    } catch (const std::runtime_error& error) {
        REQUIRE(std::string(error.what()).find("Accepted plan verifier failed:") != std::string::npos);
        PG_RETURN_VOID();
    }
    elog(ERROR, "expected accepted plan verifier to throw");
    PG_RETURN_VOID();
}
