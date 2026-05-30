extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "nodes/primnodes.h"
#include "catalog/pg_type.h"
}

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "pgx-lower/test/pgx_test_fn.h"

namespace postgresql_ast {
auto translate_const(Const* const_node, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;
} // namespace postgresql_ast

#define REQUIRE(cond) \
    do { if (!(cond)) elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond); } while (0)

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

    static Const make_const(Oid oid, int32_t typmod, Datum value, bool is_null = false) {
        Const c{};
        c.xpr.type = T_Const;
        c.consttype = oid;
        c.consttypmod = typmod;
        c.constvalue = value;
        c.constisnull = is_null;
        c.constbyval = true;
        c.constlen = -1;
        return c;
    }
};

}  // namespace

PGX_TEST_FN(ast_const_int32) {
    Fixture f;
    auto c = Fixture::make_const(INT4OID, -1, Datum{42});
    mlir::Value const v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::arith::ConstantIntOp>(op));
    REQUIRE(v.getType().isInteger(32));
    REQUIRE(mlir::cast<mlir::arith::ConstantIntOp>(op).value() == 42);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_int64) {
    Fixture f;
    auto c = Fixture::make_const(INT8OID, -1, Datum{123456789012LL});
    mlir::Value const v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(v.getType().isInteger(64));
    auto op = mlir::dyn_cast<mlir::arith::ConstantIntOp>(v.getDefiningOp());
    REQUIRE(op);
    REQUIRE(op.value() == 123456789012LL);
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_bool) {
    Fixture f;
    auto c = Fixture::make_const(BOOLOID, -1, Datum{1});
    mlir::Value const v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(v.getType().isInteger(1));
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_null) {
    Fixture f;
    auto c = Fixture::make_const(INT4OID, -1, Datum{0}, true);
    mlir::Value const v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::NullOp>(op));
    PG_RETURN_VOID();
}

PGX_TEST_FN(ast_const_date) {
    Fixture f;
    auto c = Fixture::make_const(DATEOID, -1, Datum{1234});
    mlir::Value const v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(op));
    PG_RETURN_VOID();
}
