extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
}

#include "pgx-lower/frontend/SQL/translation/translator_internals.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"

#include "mlir/IR/MLIRContext.h"

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        const auto _a = static_cast<uint32_t>(actual);                                                                 \
        const auto _e = static_cast<uint32_t>(expected);                                                               \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        const auto _a = static_cast<int32_t>(actual);                                                                  \
        const auto _e = static_cast<int32_t>(expected);                                                                \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

namespace {

struct Fixture {
    mlir::MLIRContext ctx;

    Fixture() {
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::relalg::RelAlgDialect>();
    }
};

void requirePgIdentity(mlir::Type type, mlir::db::PgOid oid, int32_t typmod, mlir::db::PgOid collation,
                       mlir::db::PgNullability nullability) {
    REQUIRE(mlir::db::isPgValueType(type));
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(type), oid);
    REQUIRE_EQ_I32(mlir::db::getPgTypmod(type), typmod);
    REQUIRE_EQ_U32(mlir::db::getPgCollation(type), collation);
    REQUIRE(mlir::db::getPgNullability(type) == nullability);
}

} // namespace

PGX_TEST_FN(pg_relalg_nullable_preserves_pg_metadata) {
    Fixture f;
    auto varchar = mlir::db::PgVarcharType::get(&f.ctx, 14, 777);
    auto nullable = mlir::db::withPgNullability(varchar, mlir::db::PgNullability::Maybe);
    requirePgIdentity(nullable, VARCHAROID, 14, 777, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_relalg_nullability_helper_recognizes_pg_semantic_nullable) {
    Fixture f;
    auto nonNull = mlir::db::PgNumericType::get(&f.ctx, 786438);
    auto nullable = mlir::db::PgNumericType::get(&f.ctx, 786438, mlir::db::PgNullability::Maybe);
    REQUIRE(!pgx_lower::frontend::sql::is_sql_nullable_type(nonNull));
    REQUIRE(pgx_lower::frontend::sql::is_sql_nullable_type(nullable));
    PG_RETURN_VOID();
}
