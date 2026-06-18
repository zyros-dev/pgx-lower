extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "nodes/plannodes.h"
}

#include "lingodb/mlir/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

// Test translator internals directly without widening the production API.
#define private public
#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#undef private

#include "pgx-lower/frontend/SQL/translation/translator_internals.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <stdexcept>
#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

struct TranslatorFixture {
    mlir::MLIRContext ctx;
    mlir::ModuleOp module;
    mlir::OpBuilder builder{&ctx};
    postgresql_ast::PostgreSQLASTTranslator::Impl translator{ctx};

    TranslatorFixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::relalg::RelAlgDialect>();
        ctx.loadDialect<mlir::scf::SCFDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
        builder.setInsertionPointToStart(module.getBody());
    }

    [[nodiscard]] auto makeContext() -> pgx_lower::frontend::sql::TranslationContext {
        auto stmt = PlannedStmt{};
        return pgx_lower::frontend::sql::TranslationContext{.current_stmt = stmt,
                                                            .builder = builder,
                                                            .current_module = module,
                                                            .current_tuple = mlir::Value{},
                                                            .outer_tuple = mlir::Value{}};
    }
};

struct DsaFixture {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder{&ctx};

    DsaFixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::dsa::DSADialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::scf::SCFDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

auto runDsaToStd(mlir::MLIRContext& ctx, mlir::ModuleOp module) -> bool {
    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::dsa::createLowerToStdPass());
    return mlir::succeeded(pm.run(module)) && mlir::succeeded(mlir::verify(module));
}

auto moduleToString(mlir::ModuleOp module) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    stream.flush();
    return output;
}

} // namespace

PGX_TEST_FN(cleanup_sweep_translator_unsupported_plan_node_throws) {
    TranslatorFixture f;
    auto ctx = f.makeContext();
    auto plan = Plan{};
    plan.type = T_Invalid;

    bool threw = false;
    try {
        (void)f.translator.translate_plan_node(ctx, &plan);
    } catch (const std::runtime_error& error) {
        threw = true;
        REQUIRE(std::string(error.what()).find("Unsupported plan node type") != std::string::npos);
    }

    REQUIRE(threw);
    PG_RETURN_VOID();
}

PGX_TEST_FN(cleanup_sweep_pgsort_free_lowers_destroy_call) {
    DsaFixture f;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&f.ctx));
    f.builder.setInsertionPointToStart(module.getBody());
    auto func = f.builder.create<mlir::func::FuncOp>(f.builder.getUnknownLoc(), "free_pgsort",
                                                     f.builder.getFunctionType({}, {}));
    auto* entry = func.addEntryBlock();
    f.builder.setInsertionPointToStart(entry);

    auto sort = f.builder.create<mlir::dsa::CreateDS>(
        f.builder.getUnknownLoc(),
        mlir::dsa::GenericIterableType::get(&f.ctx, mlir::TupleType::get(&f.ctx, {f.builder.getI32Type()}),
                                            "pgsort_iterator"));
    f.builder.create<mlir::dsa::FreeOp>(f.builder.getUnknownLoc(), sort.getDs());
    f.builder.create<mlir::func::ReturnOp>(f.builder.getUnknownLoc());

    REQUIRE(runDsaToStd(f.ctx, module));
    const auto mlir = moduleToString(module);
    REQUIRE(mlir.find("PgSortState") != std::string::npos);
    REQUIRE(mlir.find("destroy") != std::string::npos);
    PG_RETURN_VOID();
}
