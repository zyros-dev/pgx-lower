extern "C" {
#include "postgres.h"
#include "fmgr.h"
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

constexpr auto kRelid = 32970941;

auto rowOutputRelAlgMLIR(llvm::StringRef materializeRefs, llvm::StringRef materializeNames) -> std::string {
    return R"mlir(
module attributes {pgx_lower.lower_path = "row"} {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable {column_order = ["a", "b", "c"], table_identifier = "row_output_materialization|oid:)mlir"
           + std::to_string(kRelid)
           + R"mlir("} columns: {a => @row_output_materialization::@a({type = !db.pg_int4<nullable>}), b => @row_output_materialization::@b({type = !db.pg_int8}), c => @row_output_materialization::@c({type = !db.pg_int4<nullable>})}
    %1 = relalg.materialize %0 )mlir"
           + materializeRefs.str() + " => " + materializeNames.str() + R"mlir( : !dsa.table
    return %1 : !dsa.table
  }
}
)mlir";
}

auto rowOutputComputedRelAlgMLIR() -> std::string {
    return R"mlir(
module attributes {pgx_lower.lower_path = "row"} {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable {column_order = ["a", "b", "c"], table_identifier = "row_output_materialization|oid:32970941"} columns: {a => @row_output_materialization::@a({type = !db.pg_int4<nullable>}), b => @row_output_materialization::@b({type = !db.pg_int8}), c => @row_output_materialization::@c({type = !db.pg_int4<nullable>})}
    %1 = relalg.map %0 computes : [@computed::@a_is_null({type = !db.pg_bool})] (%arg0: !relalg.tuple) {
      %2 = relalg.getcol %arg0 @row_output_materialization::@a : !db.pg_int4<nullable>
      %3 = db.isnull %2 : !db.pg_int4<nullable> -> !db.pg_bool
      relalg.return %3 : !db.pg_bool
    }
    %4 = relalg.materialize %1 [@computed::@a_is_null] => ["a_is_null"] : !dsa.table
    return %4 : !dsa.table
  }
}
)mlir";
}

auto runPhase3a(llvm::StringRef mlirText) -> std::unique_ptr<pgx_test::StandalonePipelineTester> {
    auto tester = std::make_unique<pgx_test::StandalonePipelineTester>();
    REQUIRE(tester->loadRelAlgModule(mlirText.str()));
    REQUIRE(tester->runPhase3a());
    REQUIRE(tester->verifyCurrentModule());
    return tester;
}

auto singleEmitSchema(mlir::ModuleOp module) -> mlir::db::PgRowSchemaAttr {
    std::vector<mlir::db::PgEmitRowOp> emits;
    module.walk([&](mlir::db::PgEmitRowOp emit) { emits.push_back(emit); });
    REQUIRE(emits.size() == 1);
    REQUIRE(emits[0].getValues().size() == emits[0].getSchema().getFields().size());
    return emits[0].getSchema();
}

void requireNoLegacyMaterialization(mlir::ModuleOp module) {
    bool hasLegacyAppend = false;
    bool hasLegacyNextRow = false;
    module.walk([&](mlir::dsa::Append) { hasLegacyAppend = true; });
    module.walk([&](mlir::dsa::NextRow) { hasLegacyNextRow = true; });
    REQUIRE(!hasLegacyAppend);
    REQUIRE(!hasLegacyNextRow);
}

} // namespace

PGX_TEST_FN(pg_row_output_schema_targetlist_order) {
    auto tester = runPhase3a(
        rowOutputRelAlgMLIR("[@row_output_materialization::@b,@row_output_materialization::@a]", R"(["b", "a"])"));
    const auto schema = singleEmitSchema(tester->getModule());
    const auto fields = schema.getFields();

    REQUIRE_EQ_U32(fields.size(), 2);
    REQUIRE(fields[0].getName().getValue() == "b");
    REQUIRE_EQ_U32(fields[0].getOid(), INT8OID);
    REQUIRE(fields[0].getOrigin() == mlir::db::PgRowFieldOrigin::base);
    REQUIRE_EQ_I32(fields[0].getAttno(), 2);
    REQUIRE(fields[1].getName().getValue() == "a");
    REQUIRE_EQ_U32(fields[1].getOid(), INT4OID);
    REQUIRE(fields[1].getNullability() == mlir::db::PgNullability::Maybe);
    REQUIRE_EQ_I32(fields[1].getAttno(), 1);
    requireNoLegacyMaterialization(tester->getModule());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_output_duplicate_names) {
    auto tester = runPhase3a(
        rowOutputRelAlgMLIR("[@row_output_materialization::@a,@row_output_materialization::@c]", R"(["dup", "dup"])"));
    const auto schema = singleEmitSchema(tester->getModule());
    const auto fields = schema.getFields();

    REQUIRE_EQ_U32(fields.size(), 2);
    REQUIRE(fields[0].getName().getValue() == "dup");
    REQUIRE(fields[1].getName().getValue() == "dup");
    REQUIRE_EQ_I32(fields[0].getAttno(), 1);
    REQUIRE_EQ_I32(fields[1].getAttno(), 3);
    requireNoLegacyMaterialization(tester->getModule());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_output_computed_field_metadata) {
    auto tester = runPhase3a(rowOutputComputedRelAlgMLIR());
    const auto schema = singleEmitSchema(tester->getModule());
    const auto fields = schema.getFields();

    REQUIRE_EQ_U32(fields.size(), 1);
    REQUIRE(fields[0].getName().getValue() == "a_is_null");
    REQUIRE_EQ_U32(fields[0].getRelid(), InvalidOid);
    REQUIRE_EQ_I32(fields[0].getAttno(), 0);
    REQUIRE_EQ_U32(fields[0].getOid(), BOOLOID);
    REQUIRE(fields[0].getNullability() == mlir::db::PgNullability::Never);
    REQUIRE(fields[0].getOrigin() == mlir::db::PgRowFieldOrigin::computed);
    requireNoLegacyMaterialization(tester->getModule());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_output_resjunk_omitted_from_emit_schema) {
    auto tester = runPhase3a(rowOutputRelAlgMLIR("[@row_output_materialization::@a]", "[\"a\"]"));
    const auto schema = singleEmitSchema(tester->getModule());
    const auto fields = schema.getFields();

    REQUIRE_EQ_U32(fields.size(), 1);
    REQUIRE(fields[0].getName().getValue() == "a");
    REQUIRE(!fields[0].getResjunk());
    requireNoLegacyMaterialization(tester->getModule());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_output_null_flags_survive_full_lowering) {
    auto tester = runPhase3a(
        rowOutputRelAlgMLIR("[@row_output_materialization::@a,@row_output_materialization::@c]", R"(["a", "c"])"));
    REQUIRE(tester->runPhase3b());
    const auto mlir = tester->getCurrentMLIR();

    REQUIRE(mlir.find("_ZN7runtime12PgRowRuntime9emitInt32Eibiiiib") != std::string::npos);
    REQUIRE(mlir.find("_ZN7runtime12PgRowRuntime11emitRowDoneEi") != std::string::npos);
    REQUIRE(mlir.find("TableBuilder8addInt32") == std::string::npos);
    PG_RETURN_VOID();
}
