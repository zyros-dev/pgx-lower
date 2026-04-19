// Smoke test for lingodb::utility::mlir_type_to_pg_oid — a pure function
// that maps MLIR types to PG OIDs using compile-time PG #define constants
// (BOOLOID, INT4OID, etc.), no PG runtime symbols.
//
// This test exists primarily as an existence proof: yes, pure-computation
// functions in the lingodb_utility library are unit-testable. It also
// documents the type-mapping contract enough that a later spec changing
// it (e.g. adding a new dialect type) has to touch this file.

#include <gtest/gtest.h>

#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilTypes.h"
#include "lingodb/utility/mlir_to_postgres.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
}

class TypeMappingTest : public ::testing::Test {
protected:
    mlir::MLIRContext ctx;
    void SetUp() override {
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

TEST_F(TypeMappingTest, MapsBoolToPgBoolOid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx, 1)), BOOLOID);
}

TEST_F(TypeMappingTest, MapsInt16ToPgInt2Oid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx, 16)), INT2OID);
}

TEST_F(TypeMappingTest, MapsInt32ToPgInt4Oid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx, 32)), INT4OID);
}

TEST_F(TypeMappingTest, MapsInt64ToPgInt8Oid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx, 64)), INT8OID);
}

TEST_F(TypeMappingTest, MapsInt128ToPgNumericOid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::IntegerType::get(&ctx, 128)), NUMERICOID);
}

TEST_F(TypeMappingTest, MapsF32ToPgFloat4Oid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float32Type::get(&ctx)), FLOAT4OID);
}

TEST_F(TypeMappingTest, MapsF64ToPgFloat8Oid) {
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(mlir::Float64Type::get(&ctx)), FLOAT8OID);
}

TEST_F(TypeMappingTest, MapsVarLen32ToPgTextOid) {
    auto varlen = mlir::util::VarLen32Type::get(&ctx);
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(varlen), TEXTOID);
}

TEST_F(TypeMappingTest, NullableIntegerUnwrapsToInnerOid) {
    // The translator wraps nullable columns in a TupleType<i1, inner>.
    // mlir_type_to_pg_oid should look through to the inner type.
    mlir::OpBuilder builder(&ctx);
    auto i1 = builder.getI1Type();
    auto i32 = builder.getI32Type();
    auto nullable = mlir::TupleType::get(&ctx, {i1, i32});
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(nullable), INT4OID);
}

TEST_F(TypeMappingTest, UnsupportedTypeReturnsInvalidOid) {
    // Odd bit-widths have no PG mapping.
    auto weird = mlir::IntegerType::get(&ctx, 7);
    EXPECT_EQ(lingodb::utility::mlir_type_to_pg_oid(weird), InvalidOid);
}
