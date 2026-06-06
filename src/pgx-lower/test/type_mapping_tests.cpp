extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilTypes.h"
#include "lingodb/utility/mlir_to_postgres.h"
#include "pgx-lower/frontend/SQL/translation/translator_internals.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "pgx-lower/test/pgx_test_fn.h"

#define EXPECT_OID_EQ(actual, expected) \
    do { \
        Oid _a = (actual); Oid _e = (expected); \
        if (_a != _e) elog(ERROR, "%s:%d expected oid %u got %u", \
            __FILE__, __LINE__, (unsigned) _e, (unsigned) _a); \
    } while (0)

namespace {

mlir::MLIRContext& ctx() {
    static mlir::MLIRContext c;
    static bool initialized{};
    if (!initialized) {
        c.loadDialect<mlir::db::DBDialect>();
        c.loadDialect<mlir::util::UtilDialect>();
        initialized = true;
    }
    return c;
}

}  // namespace

PGX_TEST_FN(type_mapping_raw_i1_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 1)), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_raw_i16_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 16)), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_raw_i32_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 32)), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_raw_i64_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 64)), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_int128_is_not_numeric) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 128)), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_raw_f32_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float32Type::get(&ctx())), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_raw_f64_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float64Type::get(&ctx())), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_raw_varlen_is_not_pg_identity) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::util::VarLen32Type::get(&ctx())), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_nullable_physical_tuple_is_not_pg_identity) {
    mlir::OpBuilder builder(&ctx());
    auto nullable = mlir::TupleType::get(&ctx(), {builder.getI1Type(), builder.getI32Type()});
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(nullable), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_unsupported) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 7)), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_frontend_reverse_mapper_rejects_raw_carriers) {
    EXPECT_OID_EQ(postgresql_ast::PostgreSQLTypeMapper::map_mlir_type_to_oid(mlir::IntegerType::get(&ctx(), 32)),
                  InvalidOid);
    EXPECT_OID_EQ(postgresql_ast::PostgreSQLTypeMapper::map_mlir_type_to_oid(mlir::Float64Type::get(&ctx())), InvalidOid);
    EXPECT_OID_EQ(postgresql_ast::PostgreSQLTypeMapper::map_mlir_type_to_oid(mlir::db::DecimalType::get(&ctx(), 10, 2)),
                  InvalidOid);
    EXPECT_OID_EQ(postgresql_ast::PostgreSQLTypeMapper::map_mlir_type_to_oid(mlir::db::NullableType::get(
                      &ctx(), mlir::db::PgInt4Type::get(&ctx(), mlir::db::PgNullability::Maybe))),
                  InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_pg_semantic_int8) {
    auto type = mlir::db::PgInt8Type::get(&ctx());
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(type), INT8OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_pg_semantic_numeric) {
    auto type = mlir::db::PgNumericType::get(&ctx(), -1);
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(type), NUMERICOID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_pg_semantic_date) {
    auto type = mlir::db::PgDateType::get(&ctx());
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(type), DATEOID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_pg_semantic_strings_keep_distinct_oids) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::db::PgTextType::get(&ctx(), DEFAULT_COLLATION_OID)),
                  TEXTOID);
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::db::PgVarcharType::get(&ctx(), -1, DEFAULT_COLLATION_OID)),
                  VARCHAROID);
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::db::PgBpcharType::get(&ctx(), -1, DEFAULT_COLLATION_OID)),
                  BPCHAROID);
    PG_RETURN_VOID();
}
