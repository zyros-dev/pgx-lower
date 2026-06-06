extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "pgx-lower/frontend/SQL/translation/translator_internals.h"

#include "pgx-lower/test/pgx_test_fn.h"

#include <cstdint>
#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        const std::uint32_t _a = static_cast<std::uint32_t>(actual);                                                   \
        const std::uint32_t _e = static_cast<std::uint32_t>(expected);                                                 \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        const std::int32_t _a = static_cast<std::int32_t>(actual);                                                     \
        const std::int32_t _e = static_cast<std::int32_t>(expected);                                                   \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

namespace {

constexpr auto kTypmodUnconstrained = -1;
constexpr auto kNumericTypmod = 786438;
constexpr auto kVarcharTypmod = 14;
constexpr auto kBpcharTypmod = 8;
constexpr auto kDefaultCollation = 100;
constexpr auto kExplicitCollation = 777;

struct Fixture {
    mlir::MLIRContext ctx;
    postgresql_ast::PostgreSQLTypeMapper mapper{ctx};

    Fixture() {
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

auto typeToString(mlir::Type type) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    type.print(stream);
    return output;
}

void requireEqString(const std::string& actual, llvm::StringRef expected) {
    if (actual != expected) {
        elog(ERROR, "%s:%d expected '%s' got '%s'", __FILE__, __LINE__, expected.str().c_str(), actual.c_str());
    }
}

void assertPgIdentity(mlir::Type type, llvm::StringRef expectedSpelling, mlir::db::PgOid expectedOid,
                      std::int32_t expectedTypmod, mlir::db::PgOid expectedCollation,
                      mlir::db::PgNullability expectedNullability) {
    requireEqString(typeToString(type), expectedSpelling);
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(type), expectedOid);
    REQUIRE_EQ_I32(mlir::db::getPgTypmod(type), expectedTypmod);
    REQUIRE_EQ_U32(mlir::db::getPgCollation(type), expectedCollation);
    REQUIRE(mlir::db::getPgNullability(type) == expectedNullability);
}

} // namespace

PGX_TEST_FN(postgresql_type_mapper_maps_supported_pg_types) {
    Fixture f;

    assertPgIdentity(f.mapper.map_postgre_sqltype(BOOLOID, kTypmodUnconstrained, InvalidOid), "!db.pg_bool", BOOLOID,
                     kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(INT2OID, kTypmodUnconstrained, InvalidOid), "!db.pg_int2", INT2OID,
                     kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(INT4OID, kTypmodUnconstrained, InvalidOid), "!db.pg_int4", INT4OID,
                     kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(INT8OID, kTypmodUnconstrained, InvalidOid), "!db.pg_int8", INT8OID,
                     kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(FLOAT4OID, kTypmodUnconstrained, InvalidOid), "!db.pg_float4",
                     FLOAT4OID, kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(FLOAT8OID, kTypmodUnconstrained, InvalidOid), "!db.pg_float8",
                     FLOAT8OID, kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(NUMERICOID, kNumericTypmod, InvalidOid),
                     "!db.pg_numeric<typmod = 786438>", NUMERICOID, kNumericTypmod, InvalidOid,
                     mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(DATEOID, kTypmodUnconstrained, InvalidOid), "!db.pg_date", DATEOID,
                     kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(TIMESTAMPOID, kTypmodUnconstrained, InvalidOid),
                     "!db.pg_timestamp<typmod = -1>", TIMESTAMPOID, kTypmodUnconstrained, InvalidOid,
                     mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(INTERVALOID, kTypmodUnconstrained, InvalidOid),
                     "!db.pg_interval<typmod = -1>", INTERVALOID, kTypmodUnconstrained, InvalidOid,
                     mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(TEXTOID, kTypmodUnconstrained, kDefaultCollation),
                     "!db.pg_text<collation = 100>", TEXTOID, kTypmodUnconstrained, kDefaultCollation,
                     mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(VARCHAROID, kVarcharTypmod, kDefaultCollation),
                     "!db.pg_varchar<typmod = 14, collation = 100>", VARCHAROID, kVarcharTypmod, kDefaultCollation,
                     mlir::db::PgNullability::Never);
    assertPgIdentity(f.mapper.map_postgre_sqltype(BPCHAROID, kBpcharTypmod, kDefaultCollation),
                     "!db.pg_bpchar<typmod = 8, collation = 100>", BPCHAROID, kBpcharTypmod, kDefaultCollation,
                     mlir::db::PgNullability::Never);

    PG_RETURN_VOID();
}

PGX_TEST_FN(postgresql_type_mapper_preserves_nullability_and_collation) {
    Fixture f;

    assertPgIdentity(f.mapper.map_postgre_sqltype(INT4OID, kTypmodUnconstrained, InvalidOid, true),
                     "!db.pg_int4<nullable>", INT4OID, kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Maybe);
    assertPgIdentity(f.mapper.map_postgre_sqltype(TEXTOID, kTypmodUnconstrained, kExplicitCollation, true),
                     "!db.pg_text<collation = 777, nullable>", TEXTOID, kTypmodUnconstrained, kExplicitCollation,
                     mlir::db::PgNullability::Maybe);
    assertPgIdentity(f.mapper.map_postgre_sqltype(VARCHAROID, kVarcharTypmod, kExplicitCollation, true),
                     "!db.pg_varchar<typmod = 14, collation = 777, nullable>", VARCHAROID, kVarcharTypmod,
                     kExplicitCollation, mlir::db::PgNullability::Maybe);
    assertPgIdentity(f.mapper.map_postgre_sqltype(BPCHAROID, kBpcharTypmod, kExplicitCollation, true),
                     "!db.pg_bpchar<typmod = 8, collation = 777, nullable>", BPCHAROID, kBpcharTypmod,
                     kExplicitCollation, mlir::db::PgNullability::Maybe);

    PG_RETURN_VOID();
}

PGX_TEST_FN(postgresql_type_mapper_metadata_structs_carry_collation) {
    Fixture f;
    mlir::Type textType = f.mapper.map_postgre_sqltype(TEXTOID, kTypmodUnconstrained, kExplicitCollation, true);

    pgx_lower::frontend::sql::ColumnInfo columnInfo("body", TEXTOID, kTypmodUnconstrained, kExplicitCollation, true);
    REQUIRE_EQ_U32(columnInfo.type_oid, TEXTOID);
    REQUIRE_EQ_I32(columnInfo.typmod, kTypmodUnconstrained);
    REQUIRE_EQ_U32(columnInfo.collation, kExplicitCollation);
    REQUIRE(columnInfo.nullable);

    pgx_lower::frontend::sql::TranslationResult::ColumnSchema columnSchema{.table_name = "docs",
                                                                           .column_name = "body",
                                                                           .type_oid = TEXTOID,
                                                                           .typmod = kTypmodUnconstrained,
                                                                           .collation = kExplicitCollation,
                                                                           .mlir_type = textType,
                                                                           .nullable = true};
    REQUIRE_EQ_U32(columnSchema.collation, kExplicitCollation);
    REQUIRE(columnSchema.mlir_type == textType);

    pgx_lower::frontend::sql::ResolvedParam param{.table_name = "docs",
                                                  .column_name = "body",
                                                  .type_oid = TEXTOID,
                                                  .typmod = kTypmodUnconstrained,
                                                  .collation = kExplicitCollation,
                                                  .nullable = true,
                                                  .mlir_type = textType};
    REQUIRE_EQ_U32(param.collation, kExplicitCollation);
    REQUIRE(param.mlir_type == textType);

    PG_RETURN_VOID();
}
