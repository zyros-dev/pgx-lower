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

extern "C" {

PG_FUNCTION_INFO_V1(ts_test_type_mapping_bool);
Datum ts_test_type_mapping_bool(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 1)), BOOLOID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_int16);
Datum ts_test_type_mapping_int16(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 16)), INT2OID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_int32);
Datum ts_test_type_mapping_int32(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 32)), INT4OID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_int64);
Datum ts_test_type_mapping_int64(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 64)), INT8OID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_int128);
Datum ts_test_type_mapping_int128(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 128)), NUMERICOID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_f32);
Datum ts_test_type_mapping_f32(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float32Type::get(&ctx())), FLOAT4OID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_f64);
Datum ts_test_type_mapping_f64(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float64Type::get(&ctx())), FLOAT8OID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_varlen);
Datum ts_test_type_mapping_varlen(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::util::VarLen32Type::get(&ctx())), TEXTOID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_nullable_unwrap);
Datum ts_test_type_mapping_nullable_unwrap(PG_FUNCTION_ARGS) {
    mlir::OpBuilder builder(&ctx());
    auto nullable = mlir::TupleType::get(&ctx(), {builder.getI1Type(), builder.getI32Type()});
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(nullable), INT4OID);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_type_mapping_unsupported);
Datum ts_test_type_mapping_unsupported(PG_FUNCTION_ARGS) {
    EXPECT_OID_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx(), 7)), InvalidOid);
    PG_RETURN_VOID();
}

}  // extern "C"
