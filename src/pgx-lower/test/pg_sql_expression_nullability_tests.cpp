extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "nodes/primnodes.h"
#include "utils/builtins.h"
}

#define private public
#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#undef private

#include "pgx-lower/frontend/SQL/translation/translator_internals.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

constexpr Oid kInt4EqOperator = 96;
constexpr auto kTypmodUnconstrained = -1;

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::ModuleOp module;
    mlir::OpBuilder builder{&ctx};
    postgresql_ast::PostgreSQLASTTranslator::Impl translator{ctx};

    Fixture() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::relalg::RelAlgDialect>();
        ctx.loadDialect<mlir::scf::SCFDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
        builder.setInsertionPointToStart(module.getBody());
    }

    auto makeContext() -> pgx_lower::frontend::sql::TranslationContext {
        auto stmt = PlannedStmt{};
        return pgx_lower::frontend::sql::TranslationContext{.current_stmt = stmt,
                                                            .builder = builder,
                                                            .current_module = module,
                                                            .current_tuple = mlir::Value{},
                                                            .outer_tuple = mlir::Value{}};
    }

    void startFunction(llvm::StringRef name) {
        auto fn = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name, builder.getFunctionType({}, {}));
        auto* entry = fn.addEntryBlock();
        builder.setInsertionPointToStart(entry);
    }
};

auto makeBoolConst(const bool value, const bool isNull = false) -> Const {
    auto c = Const{};
    c.xpr.type = T_Const;
    c.consttype = BOOLOID;
    c.consttypmod = kTypmodUnconstrained;
    c.constcollid = InvalidOid;
    c.constvalue = isNull ? Datum{0} : BoolGetDatum(value);
    c.constisnull = isNull;
    c.constbyval = true;
    c.constlen = sizeof(bool);
    return c;
}

auto makeIntConst(const int32_t value, const bool isNull = false) -> Const {
    auto c = Const{};
    c.xpr.type = T_Const;
    c.consttype = INT4OID;
    c.consttypmod = kTypmodUnconstrained;
    c.constcollid = InvalidOid;
    c.constvalue = isNull ? Datum{0} : Int32GetDatum(value);
    c.constisnull = isNull;
    c.constbyval = true;
    c.constlen = sizeof(int32);
    return c;
}

auto makeTextConst(const char* value, const bool isNull = false) -> Const {
    auto c = Const{};
    c.xpr.type = T_Const;
    c.consttype = TEXTOID;
    c.consttypmod = kTypmodUnconstrained;
    c.constcollid = DEFAULT_COLLATION_OID;
    c.constvalue = isNull ? Datum{0} : CStringGetTextDatum(value);
    c.constisnull = isNull;
    c.constbyval = false;
    c.constlen = -1;
    return c;
}

void requirePgIdentity(mlir::Type type, mlir::db::PgOid oid, int32_t typmod, mlir::db::PgOid collation,
                       mlir::db::PgNullability nullability) {
    REQUIRE(mlir::db::isPgValueType(type));
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(type), oid);
    REQUIRE_EQ_I32(mlir::db::getPgTypmod(type), typmod);
    REQUIRE_EQ_U32(mlir::db::getPgCollation(type), collation);
    REQUIRE(mlir::db::getPgNullability(type) == nullability);
}

} // namespace

PGX_TEST_FN(pg_sql_expression_coalesce_non_null_fallback_is_not_nullable) {
    Fixture f;
    f.startFunction("coalesce_sql_nullability");
    auto ctx = f.makeContext();
    auto nullableArg = makeTextConst("ignored", true);
    auto fallbackArg = makeTextConst("fallback");
    auto coalesce = CoalesceExpr{};
    coalesce.xpr.type = T_CoalesceExpr;
    coalesce.coalescetype = TEXTOID;
    coalesce.coalescecollid = DEFAULT_COLLATION_OID;
    coalesce.args = list_make2(&nullableArg, &fallbackArg);

    const mlir::Value value = f.translator.translate_expression(ctx, reinterpret_cast<Expr*>(&coalesce));
    REQUIRE(value);
    requirePgIdentity(value.getType(), TEXTOID, kTypmodUnconstrained, DEFAULT_COLLATION_OID,
                      mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_sql_expression_bool_expr_preserves_sql_nullable_bool) {
    Fixture f;
    auto ctx = f.makeContext();
    auto trueArg = makeBoolConst(true);
    auto nullArg = makeBoolConst(false, true);
    auto boolExpr = BoolExpr{};
    boolExpr.xpr.type = T_BoolExpr;
    boolExpr.boolop = AND_EXPR;
    boolExpr.args = list_make2(&trueArg, &nullArg);

    const mlir::Value value = f.translator.translate_expression(ctx, reinterpret_cast<Expr*>(&boolExpr));
    REQUIRE(value);
    requirePgIdentity(value.getType(), BOOLOID, kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_sql_expression_null_test_returns_non_null_pg_bool) {
    Fixture f;
    auto ctx = f.makeContext();
    auto arg = makeIntConst(0, true);
    auto nullTest = NullTest{};
    nullTest.xpr.type = T_NullTest;
    nullTest.arg = reinterpret_cast<Expr*>(&arg);
    nullTest.nulltesttype = IS_NULL;

    const mlir::Value value = f.translator.translate_expression(ctx, reinterpret_cast<Expr*>(&nullTest));
    REQUIRE(value);
    requirePgIdentity(value.getType(), BOOLOID, kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_sql_expression_scalar_array_returns_sql_pg_bool) {
    Fixture f;
    auto ctx = f.makeContext();
    auto lhs = makeIntConst(2);
    auto elem1 = makeIntConst(1);
    auto elem2 = makeIntConst(2);
    auto arrayExpr = ArrayExpr{};
    arrayExpr.xpr.type = T_ArrayExpr;
    arrayExpr.array_typeid = INT4ARRAYOID;
    arrayExpr.element_typeid = INT4OID;
    arrayExpr.elements = list_make2(&elem1, &elem2);

    auto scalarArray = ScalarArrayOpExpr{};
    scalarArray.xpr.type = T_ScalarArrayOpExpr;
    scalarArray.opno = kInt4EqOperator;
    scalarArray.useOr = true;
    scalarArray.args = list_make2(&lhs, &arrayExpr);

    const mlir::Value value = f.translator.translate_expression(ctx, reinterpret_cast<Expr*>(&scalarArray));
    REQUIRE(value);
    requirePgIdentity(value.getType(), BOOLOID, kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}
