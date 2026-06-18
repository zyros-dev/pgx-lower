extern "C" {
#include "postgres.h"
#include "fmgr.h"
}

#include "pgx-lower/test/pgx_test_fn.h"
#include "pgx-lower/test/standalone_mlir_runner.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

auto rowPathRelAlgMLIR() -> const char* {
    return R"mlir(
module attributes {pgx_lower.lower_path = "row"} {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable {column_order = ["id", "payload"], table_identifier = "row_first_slice|oid:32970940"} columns: {id => @row_first_slice::@id({type = !db.pg_int8}), payload => @row_first_slice::@payload({type = !db.pg_int4<nullable>})}
    %1 = relalg.materialize %0 [@row_first_slice::@id, @row_first_slice::@payload] => ["id", "payload"] : !dsa.table
    return %1 : !dsa.table
  }
}
)mlir";
}

auto runRowPathPhase3a() -> std::unique_ptr<pgx_test::StandalonePipelineTester> {
    auto tester = std::make_unique<pgx_test::StandalonePipelineTester>();
    REQUIRE(tester->loadRelAlgModule(rowPathRelAlgMLIR()));
    REQUIRE(tester->runPhase3a());
    return tester;
}

void requireContains(const std::string& haystack, llvm::StringRef needle) {
    if (haystack.find(needle.str()) == std::string::npos) {
        elog(ERROR, "%s:%d expected to find '%s' in MLIR:\n%s", __FILE__, __LINE__, needle.str().c_str(),
             haystack.c_str());
    }
}

void requireNotContains(const std::string& haystack, llvm::StringRef needle) {
    if (haystack.find(needle.str()) != std::string::npos) {
        elog(ERROR, "%s:%d expected not to find '%s' in MLIR:\n%s", __FILE__, __LINE__, needle.str().c_str(),
             haystack.c_str());
    }
}

} // namespace

PGX_TEST_FN(pg_row_first_slice_ir_uses_row_ops) {
    auto tester = runRowPathPhase3a();
    const auto mlir = tester->getCurrentMLIR();

    requireContains(mlir, "db.pg_row_get");
    requireContains(mlir, "#db.pg_row_schema");
    requireContains(mlir, "!db.pg_row");
    requireNotContains(mlir, "dsa.scan_source");
    requireNotContains(mlir, "dsa.record_batch");
    requireNotContains(mlir, "table_chunk_iterator");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_first_slice_schema_from_scan_metadata) {
    auto tester = runRowPathPhase3a();
    const auto mlir = tester->getCurrentMLIR();

    requireContains(mlir, "relid = 32970940");
    requireContains(mlir, "attno = 1");
    requireContains(mlir, "attno = 2");
    requireContains(mlir, "name = \"id\"");
    requireContains(mlir, "name = \"payload\"");
    requireContains(mlir, "oid = 20");
    requireContains(mlir, "oid = 23");
    requireContains(mlir, "nullable = maybe");
    requireContains(mlir, "type = !db.pg_int4<nullable>");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_first_slice_full_lowering_bypasses_batch_runtime) {
    auto tester = runRowPathPhase3a();
    REQUIRE(tester->runPhase3b());
    const auto mlir = tester->getCurrentMLIR();

    requireContains(mlir, "PgRowRuntime");
    requireNotContains(mlir, "DataSourceIteration::access");
    requireNotContains(mlir, "table_chunk_iterator");
    requireNotContains(mlir, "dsa.scan_source");
    requireNotContains(mlir, "dsa.at");
    PG_RETURN_VOID();
}

extern "C" bool pgx_lower_row_first_slice_runtime_tupledesc_value_null_for_testing();
extern "C" bool pgx_lower_row_first_slice_runtime_tupledesc_mismatch_for_testing();

PGX_TEST_FN(pg_row_first_slice_runtime_tupledesc_value_null) {
    REQUIRE(pgx_lower_row_first_slice_runtime_tupledesc_value_null_for_testing());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_first_slice_runtime_tupledesc_mismatch) {
    REQUIRE(pgx_lower_row_first_slice_runtime_tupledesc_mismatch_for_testing());
    PG_RETURN_VOID();
}
