extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DB/IR/RuntimeFunctions.h"

#include "mlir/IR/MLIRContext.h"

#include "pgx-lower/test/pgx_test_fn.h"

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

struct Fixture {
    mlir::MLIRContext ctx;
    std::shared_ptr<mlir::db::RuntimeFunctionRegistry> registry;

    Fixture() {
        ctx.loadDialect<mlir::db::DBDialect>();
        registry = mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(&ctx);
    }
};

auto verifyRuntime(mlir::db::RuntimeFunctionRegistry& registry, llvm::StringRef name, mlir::TypeRange args,
                   mlir::Type result) -> bool {
    return registry.verify(name.str(), args, result);
}

} // namespace

PGX_TEST_FN(pg_runtime_function_type_accepts_pg_numeric) {
    Fixture f;
    auto numeric = mlir::db::PgNumericType::get(&f.ctx, -1);
    auto constrained = mlir::db::PgNumericType::get(&f.ctx, 786438);
    REQUIRE(verifyRuntime(*f.registry, "NumericAdd", {numeric, numeric}, numeric));
    REQUIRE(verifyRuntime(*f.registry, "NumericMul", {constrained, constrained}, constrained));
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_runtime_function_type_accepts_pg_strings) {
    Fixture f;
    auto text = mlir::db::PgTextType::get(&f.ctx, DEFAULT_COLLATION_OID);
    auto varchar = mlir::db::PgVarcharType::get(&f.ctx, -1, DEFAULT_COLLATION_OID);
    REQUIRE(verifyRuntime(*f.registry, "Upper", {text}, text));
    REQUIRE(verifyRuntime(*f.registry, "Lower", {varchar}, varchar));
    REQUIRE(verifyRuntime(*f.registry, "Like", {text, text}, mlir::db::PgBoolType::get(&f.ctx)));
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_runtime_function_type_accepts_pg_date_interval) {
    Fixture f;
    auto date = mlir::db::PgDateType::get(&f.ctx);
    auto interval = mlir::db::PgIntervalType::get(&f.ctx, -1);
    REQUIRE(verifyRuntime(*f.registry, "DateAdd", {date, interval}, date));
    REQUIRE(verifyRuntime(*f.registry, "DateSubtract", {date, interval}, date));
    PG_RETURN_VOID();
}
