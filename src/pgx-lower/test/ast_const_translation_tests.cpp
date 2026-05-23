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

namespace postgresql_ast {
auto translate_const(Const* const_node, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;
}

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

extern "C" {

PG_FUNCTION_INFO_V1(ts_test_ast_const_int32);
Datum ts_test_ast_const_int32(PG_FUNCTION_ARGS) {
    Fixture f;
    auto c = Fixture::make_const(INT4OID, -1, Datum{42});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::arith::ConstantIntOp>(op));
    REQUIRE(v.getType().isInteger(32));
    REQUIRE(mlir::cast<mlir::arith::ConstantIntOp>(op).value() == 42);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_ast_const_int64);
Datum ts_test_ast_const_int64(PG_FUNCTION_ARGS) {
    Fixture f;
    auto c = Fixture::make_const(INT8OID, -1, Datum{123456789012LL});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(v.getType().isInteger(64));
    auto op = mlir::dyn_cast<mlir::arith::ConstantIntOp>(v.getDefiningOp());
    REQUIRE(op);
    REQUIRE(op.value() == 123456789012LL);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_ast_const_bool);
Datum ts_test_ast_const_bool(PG_FUNCTION_ARGS) {
    Fixture f;
    auto c = Fixture::make_const(BOOLOID, -1, Datum{1});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    REQUIRE(v.getType().isInteger(1));
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_ast_const_null);
Datum ts_test_ast_const_null(PG_FUNCTION_ARGS) {
    Fixture f;
    auto c = Fixture::make_const(INT4OID, -1, Datum{0}, true);
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::NullOp>(op));
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_ast_const_date);
Datum ts_test_ast_const_date(PG_FUNCTION_ARGS) {
    Fixture f;
    auto c = Fixture::make_const(DATEOID, -1, Datum{1234});
    mlir::Value v = postgresql_ast::translate_const(&c, f.builder, f.ctx);
    REQUIRE(v);
    auto* op = v.getDefiningOp();
    REQUIRE(op);
    REQUIRE(mlir::isa<mlir::db::ConstantOp>(op));
    PG_RETURN_VOID();
}

}  // extern "C"
