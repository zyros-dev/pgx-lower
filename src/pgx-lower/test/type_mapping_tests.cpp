extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilTypes.h"
#include "lingodb/utility/mlir_to_postgres.h"
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
    static bool initialized = false;
    if (!initialized) {
        c.loadDialect<mlir::util::UtilDialect>();
        initialized = true;
    }
    return c;
}

}  // namespace

PGX_TEST_FN(type_mapping_bool) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 1)), BOOLOID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_int16) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 16)), INT2OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_int32) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 32)), INT4OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_int64) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 64)), INT8OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_int128) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 128)), NUMERICOID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_f32) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float32Type::get(&ctx())), FLOAT4OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_f64) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float64Type::get(&ctx())), FLOAT8OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_varlen) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::util::VarLen32Type::get(&ctx())), TEXTOID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_nullable_unwrap) {
    mlir::OpBuilder builder(&ctx());
    auto nullable = mlir::TupleType::get(&ctx(), {builder.getI1Type(), builder.getI32Type()});
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(nullable), INT4OID);
    PG_RETURN_VOID();
}

PGX_TEST_FN(type_mapping_unsupported) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 7)), InvalidOid);
    PG_RETURN_VOID();
}
