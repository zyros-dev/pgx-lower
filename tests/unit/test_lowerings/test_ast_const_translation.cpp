// Smoke test for postgresql_ast::translate_const — proves the
// AST-to-MLIR translation layer is unit-testable.
//
// Input side: fabricate a `Const` node on the stack. PG's Const is a
// POD struct defined in postgres.h; we initialize the fields a test
// cares about (constisnull, consttype, consttypmod, constvalue) and
// leave the rest zero. The NUMERICOID and TIMESTAMPOID paths of
// translate_const already `#ifdef POSTGRESQL_EXTENSION`-guard their
// PG-runtime calls, so the simple-integer / float / bool / date cases
// all work in the unit-test build without needing palloc or
// DirectFunctionCall1 stubs.
//
// Output side: assert the returned MLIR Value has the expected type and
// that the producing op is the expected kind (arith.constant for
// integers/floats, db.constant for decimal strings, db.null for null).

#include <gtest/gtest.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

extern "C" {
#include "postgres.h"
#include "nodes/primnodes.h"
#include "catalog/pg_type.h"
}

// translate_const is declared in the internals header alongside a lot of
// other things we don't need here; forward-declare just what we call.
namespace postgresql_ast {
auto translate_const(Const* const_node, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value;
}  // namespace postgresql_ast

class AstConstTranslationTest : public ::testing::Test {
protected:
    mlir::MLIRContext ctx;
    mlir::ModuleOp module;
    mlir::OpBuilder builder{&ctx};

    void SetUp() override {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
        builder.setInsertionPointToStart(module.getBody());
    }

    // Build a minimal Const node. PG's Const is a C struct; POD-init
    // here is fine for test purposes (the translator only reads the
    // fields we set).
    Const make_const(Oid type_oid, int32_t typmod, Datum value, bool is_null = false) {
        Const c{};
        c.xpr.type = T_Const;
        c.consttype = type_oid;
        c.consttypmod = typmod;
        c.constvalue = value;
        c.constisnull = is_null;
        c.constbyval = true;
        c.constlen = -1;
        return c;
    }
};

TEST_F(AstConstTranslationTest, Int32_ProducesArithConstant) {
    auto c = make_const(INT4OID, -1, Datum{42});
    mlir::Value v = postgresql_ast::translate_const(&c, builder, ctx);

    ASSERT_TRUE(v);
    auto* op = v.getDefiningOp();
    ASSERT_TRUE(op);
    EXPECT_TRUE(mlir::isa<mlir::arith::ConstantIntOp>(op));
    EXPECT_TRUE(v.getType().isInteger(32));

    auto const_op = mlir::cast<mlir::arith::ConstantIntOp>(op);
    EXPECT_EQ(const_op.value(), 42);
}

TEST_F(AstConstTranslationTest, Int64_ProducesArithConstant) {
    auto c = make_const(INT8OID, -1, Datum{123456789012LL});
    mlir::Value v = postgresql_ast::translate_const(&c, builder, ctx);

    ASSERT_TRUE(v);
    EXPECT_TRUE(v.getType().isInteger(64));
    auto const_op = mlir::dyn_cast<mlir::arith::ConstantIntOp>(v.getDefiningOp());
    ASSERT_TRUE(const_op);
    EXPECT_EQ(const_op.value(), 123456789012LL);
}

TEST_F(AstConstTranslationTest, Bool_ProducesArithConstant) {
    auto c = make_const(BOOLOID, -1, Datum{1});
    mlir::Value v = postgresql_ast::translate_const(&c, builder, ctx);

    ASSERT_TRUE(v);
    // bool lowers to i1 in MLIR.
    EXPECT_TRUE(v.getType().isInteger(1));
}

TEST_F(AstConstTranslationTest, NullConst_ProducesDbNullOp) {
    // constisnull = true bypasses the value cast entirely and returns db.null.
    auto c = make_const(INT4OID, -1, Datum{0}, /*is_null=*/true);
    mlir::Value v = postgresql_ast::translate_const(&c, builder, ctx);

    ASSERT_TRUE(v);
    auto* op = v.getDefiningOp();
    ASSERT_TRUE(op);
    EXPECT_TRUE(mlir::isa<mlir::db::NullOp>(op));
}

TEST_F(AstConstTranslationTest, Date_ProducesDbConstant) {
    // DATEOID path: value is days since PG epoch.
    auto c = make_const(DATEOID, -1, Datum{1234});
    mlir::Value v = postgresql_ast::translate_const(&c, builder, ctx);

    ASSERT_TRUE(v);
    auto* op = v.getDefiningOp();
    ASSERT_TRUE(op);
    EXPECT_TRUE(mlir::isa<mlir::db::ConstantOp>(op));
}
