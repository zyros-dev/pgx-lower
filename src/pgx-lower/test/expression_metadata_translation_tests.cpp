extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "utils/builtins.h"
}

// Test expression translation directly without widening the production translator API.
#define private public
#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#undef private

#include "pgx-lower/frontend/SQL/translation/translator_internals.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

constexpr auto kTypmodUnconstrained = -1;
constexpr auto kVarcharTypmod = 14;

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::ModuleOp module;
    mlir::OpBuilder builder{&ctx};
    postgresql_ast::PostgreSQLASTTranslator::Impl translator{ctx};

    Fixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
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

    static auto makeTextConst(Oid oid, int32_t typmod, const char* value, Oid collation, bool isNull = false) -> Const {
        auto c = Const{};
        c.xpr.type = T_Const;
        c.consttype = oid;
        c.consttypmod = typmod;
        c.constcollid = collation;
        c.constvalue = isNull ? Datum{0} : CStringGetTextDatum(value);
        c.constisnull = isNull;
        c.constbyval = false;
        c.constlen = -1;
        return c;
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

PGX_TEST_FN(expression_relabel_type_preserves_pg_metadata) {
    Fixture f;
    auto ctx = f.makeContext();
    auto arg = Fixture::makeTextConst(TEXTOID, kTypmodUnconstrained, "hello", DEFAULT_COLLATION_OID, true);
    auto relabel = RelabelType{};
    relabel.xpr.type = T_RelabelType;
    relabel.arg = reinterpret_cast<Expr*>(&arg);
    relabel.resulttype = VARCHAROID;
    relabel.resulttypmod = kVarcharTypmod;
    relabel.resultcollid = DEFAULT_COLLATION_OID;
    relabel.relabelformat = COERCE_IMPLICIT_CAST;

    const mlir::Value value = f.translator.translate_expression(ctx, reinterpret_cast<Expr*>(&relabel));
    REQUIRE(value);
    REQUIRE(mlir::isa<mlir::db::CastOp>(value.getDefiningOp()));
    requirePgIdentity(value.getType(), VARCHAROID, kVarcharTypmod, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(expression_coerce_via_io_preserves_pg_result_metadata) {
    Fixture f;
    auto ctx = f.makeContext();
    auto arg = Fixture::makeTextConst(TEXTOID, kTypmodUnconstrained, "hello", DEFAULT_COLLATION_OID, true);
    auto coerce = CoerceViaIO{};
    coerce.xpr.type = T_CoerceViaIO;
    coerce.arg = reinterpret_cast<Expr*>(&arg);
    coerce.resulttype = VARCHAROID;
    coerce.resultcollid = C_COLLATION_OID;
    coerce.coerceformat = COERCE_IMPLICIT_CAST;

    const mlir::Value value = f.translator.translate_expression(ctx, reinterpret_cast<Expr*>(&coerce));
    REQUIRE(value);
    REQUIRE(mlir::isa<mlir::db::CastOp>(value.getDefiningOp()));
    requirePgIdentity(value.getType(), VARCHAROID, kTypmodUnconstrained, C_COLLATION_OID, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}
