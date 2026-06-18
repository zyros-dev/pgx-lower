extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "pgx-lower/test/pgx_test_fn.h"
#include "pgx-lower/test/standalone_mlir_runner.h"

#include <memory>
#include <string>
#include <vector>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        const auto _a = static_cast<std::uint32_t>(actual);                                                            \
        const auto _e = static_cast<std::uint32_t>(expected);                                                          \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        const auto _a = static_cast<std::int32_t>(actual);                                                             \
        const auto _e = static_cast<std::int32_t>(expected);                                                           \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

namespace {

constexpr auto kRelid = 32970942;
constexpr int32_t kNumericTypmod = 786438;
constexpr int32_t kVarcharTypmod = 20;
constexpr int32_t kBpcharTypmod = 8;

auto primitiveRelAlgMLIR() -> std::string {
    return R"mlir(
module attributes {pgx_lower.lower_path = "row"} {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable {column_order = ["flag", "small", "i4", "i8", "f4", "f8", "amount", "d", "ts", "iv", "t", "v", "c"], table_identifier = "row_primitive_surface|oid:)mlir"
           + std::to_string(kRelid) + R"mlir("} columns: {
      flag => @row_primitive_surface::@flag({type = !db.pg_bool<nullable>}),
      small => @row_primitive_surface::@small({type = !db.pg_int2<nullable>}),
      i4 => @row_primitive_surface::@i4({type = !db.pg_int4<nullable>}),
      i8 => @row_primitive_surface::@i8({type = !db.pg_int8}),
      f4 => @row_primitive_surface::@f4({type = !db.pg_float4<nullable>}),
      f8 => @row_primitive_surface::@f8({type = !db.pg_float8}),
      amount => @row_primitive_surface::@amount({type = !db.pg_numeric<typmod = 786438, nullable>}),
      d => @row_primitive_surface::@d({type = !db.pg_date<nullable>}),
      ts => @row_primitive_surface::@ts({type = !db.pg_timestamp<typmod = -1, nullable>}),
      iv => @row_primitive_surface::@iv({type = !db.pg_interval<typmod = -1, nullable>}),
      t => @row_primitive_surface::@t({type = !db.pg_text<collation = 100, nullable>}),
      v => @row_primitive_surface::@v({type = !db.pg_varchar<typmod = 20, collation = 100, nullable>}),
      c => @row_primitive_surface::@c({type = !db.pg_bpchar<typmod = 8, collation = 100, nullable>})
    }
    %1 = relalg.materialize %0 [@row_primitive_surface::@flag, @row_primitive_surface::@small, @row_primitive_surface::@i4, @row_primitive_surface::@i8, @row_primitive_surface::@f4, @row_primitive_surface::@f8, @row_primitive_surface::@amount, @row_primitive_surface::@d, @row_primitive_surface::@ts, @row_primitive_surface::@iv, @row_primitive_surface::@t, @row_primitive_surface::@v, @row_primitive_surface::@c] => ["flag", "small", "i4", "i8", "f4", "f8", "amount", "d", "ts", "iv", "t", "v", "c"] : !dsa.table
    return %1 : !dsa.table
  }
}
)mlir";
}

auto runPhase3a() -> std::unique_ptr<pgx_test::StandalonePipelineTester> {
    auto tester = std::make_unique<pgx_test::StandalonePipelineTester>();
    REQUIRE(tester->loadRelAlgModule(primitiveRelAlgMLIR()));
    REQUIRE(tester->runPhase3a());
    REQUIRE(tester->verifyCurrentModule());
    return tester;
}

void requireContains(const std::string& haystack, const char* needle) {
    if (haystack.find(needle) == std::string::npos) {
        elog(ERROR, "%s:%d expected to find '%s' in MLIR:\n%s", __FILE__, __LINE__, needle, haystack.c_str());
    }
}

void requireNotContains(const std::string& haystack, const char* needle) {
    if (haystack.find(needle) != std::string::npos) {
        elog(ERROR, "%s:%d expected not to find '%s' in MLIR:\n%s", __FILE__, __LINE__, needle, haystack.c_str());
    }
}

auto singleEmitSchema(mlir::ModuleOp module) -> mlir::db::PgRowSchemaAttr {
    std::vector<mlir::db::PgEmitRowOp> emits;
    module.walk([&](mlir::db::PgEmitRowOp emit) { emits.push_back(emit); });
    REQUIRE(emits.size() == 1);
    REQUIRE(emits[0].getValues().size() == emits[0].getSchema().getFields().size());
    return emits[0].getSchema();
}

} // namespace

PGX_TEST_FN(pg_row_primitive_schema_preserves_pg_metadata) {
    auto tester = runPhase3a();
    const auto schema = singleEmitSchema(tester->getModule());
    const auto fields = schema.getFields();

    REQUIRE_EQ_U32(fields.size(), 13);
    REQUIRE_EQ_U32(fields[0].getOid(), BOOLOID);
    REQUIRE_EQ_U32(fields[1].getOid(), INT2OID);
    REQUIRE_EQ_U32(fields[2].getOid(), INT4OID);
    REQUIRE_EQ_U32(fields[3].getOid(), INT8OID);
    REQUIRE_EQ_U32(fields[4].getOid(), FLOAT4OID);
    REQUIRE_EQ_U32(fields[5].getOid(), FLOAT8OID);
    REQUIRE_EQ_U32(fields[6].getOid(), NUMERICOID);
    REQUIRE_EQ_I32(fields[6].getTypmod(), kNumericTypmod);
    REQUIRE_EQ_U32(fields[7].getOid(), DATEOID);
    REQUIRE_EQ_U32(fields[8].getOid(), TIMESTAMPOID);
    REQUIRE_EQ_U32(fields[9].getOid(), INTERVALOID);
    REQUIRE_EQ_U32(fields[10].getOid(), TEXTOID);
    REQUIRE_EQ_U32(fields[10].getCollation(), DEFAULT_COLLATION_OID);
    REQUIRE_EQ_U32(fields[11].getOid(), VARCHAROID);
    REQUIRE_EQ_I32(fields[11].getTypmod(), kVarcharTypmod);
    REQUIRE_EQ_U32(fields[11].getCollation(), DEFAULT_COLLATION_OID);
    REQUIRE_EQ_U32(fields[12].getOid(), BPCHAROID);
    REQUIRE_EQ_I32(fields[12].getTypmod(), kBpcharTypmod);
    REQUIRE_EQ_U32(fields[12].getCollation(), DEFAULT_COLLATION_OID);
    REQUIRE(fields[0].getNullability() == mlir::db::PgNullability::Maybe);
    REQUIRE(fields[3].getNullability() == mlir::db::PgNullability::Never);
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_primitive_ir_has_no_dsa_batch) {
    auto tester = runPhase3a();
    const auto mlir = tester->getCurrentMLIR();

    requireContains(mlir, "db.pg_row_get");
    requireContains(mlir, "db.pg_emit_row");
    requireNotContains(mlir, "dsa.record_batch");
    requireNotContains(mlir, "table_chunk_iterator");
    requireNotContains(mlir, "dsa.at");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_primitive_runtime_roundtrip) {
    auto tester = runPhase3a();
    REQUIRE(tester->runPhase3b());
    const auto mlir = tester->getCurrentMLIR();

    requireContains(mlir, "PgRowRuntime");
    requireNotContains(mlir, "DataSourceIteration::access");
    requireNotContains(mlir, "TableBuilder7addBool");
    requireNotContains(mlir, "TableBuilder8addInt32");
    requireNotContains(mlir, "TableBuilder9addBinary");
    PG_RETURN_VOID();
}
