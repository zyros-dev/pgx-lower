extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "nodes/primnodes.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
}

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "pgx-lower/test/pgx_test_fn.h"

namespace postgresql_ast {
auto translate_const(Const* const_node, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;
} // namespace postgresql_ast

#define REQUIRE(cond) \
    do { if (!(cond)) elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); } while (0)

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        auto _a = static_cast<uint32_t>(actual);                                                                       \
        auto _e = static_cast<uint32_t>(expected);                                                                     \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        auto _a = static_cast<int32_t>(actual);                                                                        \
        auto _e = static_cast<int32_t>(expected);                                                                      \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I64(actual, expected)                                                                               \
    do {                                                                                                               \
        auto _a = static_cast<int64_t>(actual);                                                                        \
        auto _e = static_cast<int64_t>(expected);                                                                      \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %ld got %ld", __FILE__, __LINE__, _e, _a);                                     \
        }                                                                                                              \
    } while (0)

namespace {

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::ModuleOp module;
    mlir::OpBuilder builder{&ctx};

    Fixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
        builder.setInsertionPointToStart(module.getBody());
    }

    static Const make_const(Oid oid, int32_t typmod, Datum value, bool is_null = false, Oid collation = InvalidOid) {
        Const c{};
        c.xpr.type = T_Const;
        c.consttype = oid;
        c.consttypmod = typmod;
        c.constcollid = collation;
        c.constvalue = value;
        c.constisnull = is_null;
        c.constbyval = true;
        c.constlen = -1;
        return c;
    }

    static Const make_text_const(Oid oid, int32_t typmod, const char* value, Oid collation) {
        auto c = make_const(oid, typmod, CStringGetTextDatum(value), false, collation);
        c.constbyval = false;
        return c;
    }

    static Const make_interval_const(int64_t time, int32_t day, int32_t month) {
        auto* interval = static_cast<Interval*>(palloc(sizeof(Interval)));
        interval->time = time;
        interval->day = day;
        interval->month = month;

        auto c = make_const(INTERVALOID, -1, IntervalPGetDatum(interval));
        c.constbyval = false;
        c.constlen = sizeof(Interval);
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

PGX_TEST_FN(ast_const_int32) {
    Fixture f;
    auto c = Fixture::make_const(INT4OID, -1, Datum{42});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(op));
    requirePgIdentity(v.getType(), INT4OID, -1, InvalidOid, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_int64) {
    Fixture f;
    auto c = Fixture::make_const(INT8OID, -1, Datum{123456789012LL});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), INT8OID, -1, InvalidOid, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_bool) {
    Fixture f;
    auto c = Fixture::make_const(BOOLOID, -1, Datum{1});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), BOOLOID, -1, InvalidOid, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null_int32) {
    Fixture f;
    auto c = Fixture::make_const(INT4OID, -1, Datum{0}, true);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::NullOp>(op));
    requirePgIdentity(v.getType(), INT4OID, -1, InvalidOid, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null_int64) {
    Fixture f;
    auto c = Fixture::make_const(INT8OID, -1, Datum{0}, true);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::NullOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), INT8OID, -1, InvalidOid, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null_numeric) {
    Fixture f;
    auto c = Fixture::make_const(NUMERICOID, -1, Datum{0}, true);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::NullOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), NUMERICOID, -1, InvalidOid, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null_date) {
    Fixture f;
    auto c = Fixture::make_const(DATEOID, -1, Datum{0}, true);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::NullOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), DATEOID, -1, InvalidOid, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null_timestamp) {
    Fixture f;
    auto c = Fixture::make_const(TIMESTAMPOID, -1, Datum{0}, true);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::NullOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), TIMESTAMPOID, -1, InvalidOid, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null_text) {
    Fixture f;
    auto c = Fixture::make_const(TEXTOID, -1, Datum{0}, true, DEFAULT_COLLATION_OID);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(mlir::isa<mlir::db::NullOp>(v.getDefiningOp()));
    requirePgIdentity(v.getType(), TEXTOID, -1, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_date) {
    Fixture f;
    auto c = Fixture::make_const(DATEOID, -1, Datum{1234});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(op));
    REQUIRE(mlir::isa<mlir::db::PgDateType>(v.getType()));
    requirePgIdentity(v.getType(), DATEOID, -1, InvalidOid, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_timestamp_preserves_pg_microseconds) {
    Fixture f;
    const Timestamp timestamp = USECS_PER_DAY + 123456;
    auto c = Fixture::make_const(TIMESTAMPOID, -1, TimestampGetDatum(timestamp));
    c.constlen = sizeof(Timestamp);

    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto constant = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(v.getDefiningOp());
    REQUIRE(constant);
    REQUIRE(mlir::isa<mlir::db::PgTimestampType>(v.getType()));
    requirePgIdentity(v.getType(), TIMESTAMPOID, -1, InvalidOid, mlir::db::PgNullability::Never);

    auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constant.getConstantValue());
    REQUIRE(attr);
    REQUIRE_EQ_I64(attr.getInt(), timestamp);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_interval_preserves_day_time) {
    Fixture f;
    auto c = Fixture::make_interval_const(123456, 90, 0);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    auto constant = mlir::dyn_cast<mlir::db::ConstantOp>(op);
    REQUIRE(constant);
    requirePgIdentity(v.getType(), INTERVALOID, -1, InvalidOid, mlir::db::PgNullability::Never);

    auto tupleAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(constant.getConstantValue());
    REQUIRE(tupleAttr);
    REQUIRE(tupleAttr.size() == 3);
    REQUIRE(mlir::cast<mlir::IntegerAttr>(tupleAttr[0]).getInt() == 123456);
    REQUIRE(mlir::cast<mlir::IntegerAttr>(tupleAttr[1]).getInt() == 90);
    REQUIRE(mlir::cast<mlir::IntegerAttr>(tupleAttr[2]).getInt() == 0);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_interval_preserves_month) {
    Fixture f;
    auto c = Fixture::make_interval_const(0, 0, 1);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    auto constant = mlir::dyn_cast<mlir::db::ConstantOp>(op);
    REQUIRE(constant);
    requirePgIdentity(v.getType(), INTERVALOID, -1, InvalidOid, mlir::db::PgNullability::Never);

    auto tupleAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(constant.getConstantValue());
    REQUIRE(tupleAttr);
    REQUIRE(tupleAttr.size() == 3);
    REQUIRE(mlir::cast<mlir::IntegerAttr>(tupleAttr[0]).getInt() == 0);
    REQUIRE(mlir::cast<mlir::IntegerAttr>(tupleAttr[1]).getInt() == 0);
    REQUIRE(mlir::cast<mlir::IntegerAttr>(tupleAttr[2]).getInt() == 1);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_text_varchar_bpchar_keep_distinct_pg_types) {
    Fixture f;
    auto text = Fixture::make_text_const(TEXTOID, -1, "hello", DEFAULT_COLLATION_OID);
    auto varchar = Fixture::make_text_const(VARCHAROID, 14, "hello", DEFAULT_COLLATION_OID);
    auto bpchar = Fixture::make_text_const(BPCHAROID, 8, "hello", DEFAULT_COLLATION_OID);

    auto textValue = postgresql_ast::translate_const(&text, f.builder, f.ctx);
    auto varcharValue = postgresql_ast::translate_const(&varchar, f.builder, f.ctx);
    auto bpcharValue = postgresql_ast::translate_const(&bpchar, f.builder, f.ctx);

    REQUIRE(mlir::isa<mlir::db::ConstantOp>(textValue.getDefiningOp()));
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(varcharValue.getDefiningOp()));
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(bpcharValue.getDefiningOp()));
    requirePgIdentity(textValue.getType(), TEXTOID, -1, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Never);
    requirePgIdentity(varcharValue.getType(), VARCHAROID, 14, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Never);
    requirePgIdentity(bpcharValue.getType(), BPCHAROID, 8, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}
