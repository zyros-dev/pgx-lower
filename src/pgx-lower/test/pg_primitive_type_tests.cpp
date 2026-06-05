extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "pgx-lower/test/pgx_test_fn.h"

#include <cstdint>
#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond))                                                                                                   \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
    } while (0)

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        const std::uint32_t _a = static_cast<std::uint32_t>(actual);                                                   \
        const std::uint32_t _e = static_cast<std::uint32_t>(expected);                                                 \
        if (_a != _e)                                                                                                  \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        const std::int32_t _a = static_cast<std::int32_t>(actual);                                                     \
        const std::int32_t _e = static_cast<std::int32_t>(expected);                                                   \
        if (_a != _e)                                                                                                  \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
    } while (0)

namespace {

constexpr auto kTypmodUnconstrained = -1;
constexpr auto kNumericTypmod = 786438;
constexpr auto kVarcharTypmod = 14;
constexpr auto kBpcharTypmod = 8;
constexpr auto kDefaultCollation = 100;

struct Fixture {
    mlir::MLIRContext ctx;

    Fixture() {
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

auto typeToString(mlir::Type type) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    type.print(stream);
    return output;
}

auto moduleToString(mlir::ModuleOp module) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    return output;
}

void requireEqString(const std::string& actual, llvm::StringRef expected) {
    if (actual != expected) {
        elog(ERROR, "%s:%d expected '%s' got '%s'", __FILE__, __LINE__, expected.str().c_str(), actual.c_str());
    }
}

void requireContains(const std::string& haystack, llvm::StringRef needle) {
    if (haystack.find(needle.str()) == std::string::npos) {
        elog(ERROR, "%s:%d expected to find '%s'", __FILE__, __LINE__, needle.str().c_str());
    }
}

void assertPgIdentity(mlir::Type type, llvm::StringRef expectedSpelling, mlir::db::PgOid expectedOid,
                      std::int32_t expectedTypmod, mlir::db::PgOid expectedCollation,
                      mlir::db::PgNullability expectedNullability, llvm::StringRef expectedCarrier) {
    requireEqString(typeToString(type), expectedSpelling);
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(type), expectedOid);
    REQUIRE_EQ_I32(mlir::db::getPgTypmod(type), expectedTypmod);
    REQUIRE_EQ_U32(mlir::db::getPgCollation(type), expectedCollation);
    REQUIRE(mlir::db::getPgNullability(type) == expectedNullability);
    requireEqString(typeToString(mlir::db::getPgPhysicalCarrierType(type)), expectedCarrier);
}

} // namespace

PGX_TEST_FN(pg_primitive_type_metadata_and_carriers) {
    Fixture f;

    REQUIRE_EQ_U32(mlir::db::kPgInvalidOid, InvalidOid);
    REQUIRE_EQ_U32(mlir::db::kPgBoolOid, BOOLOID);
    REQUIRE_EQ_U32(mlir::db::kPgInt2Oid, INT2OID);
    REQUIRE_EQ_U32(mlir::db::kPgInt4Oid, INT4OID);
    REQUIRE_EQ_U32(mlir::db::kPgInt8Oid, INT8OID);
    REQUIRE_EQ_U32(mlir::db::kPgTextOid, TEXTOID);
    REQUIRE_EQ_U32(mlir::db::kPgFloat4Oid, FLOAT4OID);
    REQUIRE_EQ_U32(mlir::db::kPgFloat8Oid, FLOAT8OID);
    REQUIRE_EQ_U32(mlir::db::kPgBpcharOid, BPCHAROID);
    REQUIRE_EQ_U32(mlir::db::kPgVarcharOid, VARCHAROID);
    REQUIRE_EQ_U32(mlir::db::kPgDateOid, DATEOID);
    REQUIRE_EQ_U32(mlir::db::kPgTimestampOid, TIMESTAMPOID);
    REQUIRE_EQ_U32(mlir::db::kPgIntervalOid, INTERVALOID);
    REQUIRE_EQ_U32(mlir::db::kPgNumericOid, NUMERICOID);

    assertPgIdentity(mlir::db::PgBoolType::get(&f.ctx), "!db.pg_bool", mlir::db::kPgBoolOid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "i1");
    assertPgIdentity(mlir::db::PgInt2Type::get(&f.ctx), "!db.pg_int2", mlir::db::kPgInt2Oid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "i16");
    assertPgIdentity(mlir::db::PgInt4Type::get(&f.ctx), "!db.pg_int4", mlir::db::kPgInt4Oid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "i32");
    assertPgIdentity(mlir::db::PgInt8Type::get(&f.ctx), "!db.pg_int8", mlir::db::kPgInt8Oid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "i64");
    assertPgIdentity(mlir::db::PgFloat4Type::get(&f.ctx), "!db.pg_float4", mlir::db::kPgFloat4Oid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "f32");
    assertPgIdentity(mlir::db::PgFloat8Type::get(&f.ctx), "!db.pg_float8", mlir::db::kPgFloat8Oid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "f64");
    assertPgIdentity(mlir::db::PgNumericType::get(&f.ctx, kNumericTypmod), "!db.pg_numeric<typmod = 786438>",
                     mlir::db::kPgNumericOid, kNumericTypmod, mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never,
                     "i64");
    assertPgIdentity(mlir::db::PgDateType::get(&f.ctx), "!db.pg_date", mlir::db::kPgDateOid, kTypmodUnconstrained,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Never, "i64");
    assertPgIdentity(mlir::db::PgTimestampType::get(&f.ctx, kTypmodUnconstrained), "!db.pg_timestamp<typmod = -1>",
                     mlir::db::kPgTimestampOid, kTypmodUnconstrained, mlir::db::kPgInvalidOid,
                     mlir::db::PgNullability::Never, "i64");
    assertPgIdentity(mlir::db::PgIntervalType::get(&f.ctx, kTypmodUnconstrained), "!db.pg_interval<typmod = -1>",
                     mlir::db::kPgIntervalOid, kTypmodUnconstrained, mlir::db::kPgInvalidOid,
                     mlir::db::PgNullability::Never, "i64");
    assertPgIdentity(mlir::db::PgTextType::get(&f.ctx, kDefaultCollation), "!db.pg_text<collation = 100>",
                     mlir::db::kPgTextOid, kTypmodUnconstrained, kDefaultCollation, mlir::db::PgNullability::Never,
                     "!util.varlen32");
    assertPgIdentity(mlir::db::PgVarcharType::get(&f.ctx, kVarcharTypmod, kDefaultCollation),
                     "!db.pg_varchar<typmod = 14, collation = 100>", mlir::db::kPgVarcharOid, kVarcharTypmod,
                     kDefaultCollation, mlir::db::PgNullability::Never, "!util.varlen32");
    assertPgIdentity(mlir::db::PgBpcharType::get(&f.ctx, kBpcharTypmod, kDefaultCollation),
                     "!db.pg_bpchar<typmod = 8, collation = 100>", mlir::db::kPgBpcharOid, kBpcharTypmod,
                     kDefaultCollation, mlir::db::PgNullability::Never, "!util.varlen32");

    assertPgIdentity(mlir::db::PgInt4Type::get(&f.ctx, mlir::db::PgNullability::Maybe), "!db.pg_int4<nullable>",
                     mlir::db::kPgInt4Oid, kTypmodUnconstrained, mlir::db::kPgInvalidOid,
                     mlir::db::PgNullability::Maybe, "i32");
    assertPgIdentity(mlir::db::PgNumericType::get(&f.ctx, kNumericTypmod, mlir::db::PgNullability::Maybe),
                     "!db.pg_numeric<typmod = 786438, nullable>", mlir::db::kPgNumericOid, kNumericTypmod,
                     mlir::db::kPgInvalidOid, mlir::db::PgNullability::Maybe, "i64");

    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_primitive_type_erased_helpers_preserve_identity) {
    Fixture f;

    mlir::Type numeric = mlir::db::PgNumericType::get(&f.ctx, kNumericTypmod);
    mlir::Type nullableNumeric = mlir::db::withPgNullability(numeric, mlir::db::PgNullability::Maybe);
    assertPgIdentity(nullableNumeric, "!db.pg_numeric<typmod = 786438, nullable>", mlir::db::kPgNumericOid,
                     kNumericTypmod, mlir::db::kPgInvalidOid, mlir::db::PgNullability::Maybe, "i64");

    mlir::Type varchar = mlir::db::PgVarcharType::get(&f.ctx, kVarcharTypmod, kDefaultCollation);
    mlir::Type nullableVarchar = mlir::db::withPgNullability(varchar, mlir::db::PgNullability::Maybe);
    assertPgIdentity(nullableVarchar, "!db.pg_varchar<typmod = 14, collation = 100, nullable>", mlir::db::kPgVarcharOid,
                     kVarcharTypmod, kDefaultCollation, mlir::db::PgNullability::Maybe, "!util.varlen32");

    mlir::Type int8 = mlir::db::PgInt8Type::get(&f.ctx);
    requireEqString(typeToString(mlir::db::getPgPhysicalCarrierType(int8)), "i64");
    requireEqString(typeToString(mlir::db::getPgPhysicalCarrierType(numeric)), "i64");
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(int8), INT8OID);
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(numeric), NUMERICOID);

    mlir::Type machinery = mlir::IntegerType::get(&f.ctx, 32);
    REQUIRE(!mlir::db::isPgValueType(machinery));
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(machinery), mlir::db::kPgInvalidOid);
    REQUIRE_EQ_I32(mlir::db::getPgTypmod(machinery), kTypmodUnconstrained);
    REQUIRE_EQ_U32(mlir::db::getPgCollation(machinery), mlir::db::kPgInvalidOid);
    REQUIRE(mlir::db::withPgNullability(machinery, mlir::db::PgNullability::Maybe) == mlir::Type{});
    REQUIRE(mlir::db::getPgPhysicalCarrierType(machinery) == mlir::Type{});

    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_primitive_type_roundtrip_print_parse) {
    Fixture f;

    const char* moduleText = R"mlir(
module {
  func.func @uses_pg_types(%a: !db.pg_bool, %b: !db.pg_int4<nullable>,
                           %c: !db.pg_numeric<typmod = -1>,
                           %d: !db.pg_numeric<typmod = 786438, nullable>,
                           %e: !db.pg_text<collation = 100>,
                           %f: !db.pg_varchar<typmod = 14, collation = 100, nullable>,
                           %g: !db.pg_bpchar<typmod = 8, collation = 100>,
                           %h: !db.pg_date,
                           %i: !db.pg_timestamp<typmod = -1>,
                           %j: !db.pg_interval<typmod = -1>,
                           %k: !db.pg_float4,
                           %l: !db.pg_float8,
                           %m: !db.pg_int2,
                           %n: !db.pg_int8) {
    return
  }
}
)mlir";

    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(moduleText, &f.ctx);
    REQUIRE(module);
    const std::string printed = moduleToString(*module);
    requireContains(printed, "!db.pg_bool");
    requireContains(printed, "!db.pg_int4<nullable>");
    requireContains(printed, "!db.pg_numeric<typmod = -1>");
    requireContains(printed, "!db.pg_numeric<typmod = 786438, nullable>");
    requireContains(printed, "!db.pg_text<collation = 100>");
    requireContains(printed, "!db.pg_varchar<typmod = 14, collation = 100, nullable>");
    requireContains(printed, "!db.pg_bpchar<typmod = 8, collation = 100>");
    requireContains(printed, "!db.pg_date");
    requireContains(printed, "!db.pg_timestamp<typmod = -1>");
    requireContains(printed, "!db.pg_interval<typmod = -1>");
    requireContains(printed, "!db.pg_float4");
    requireContains(printed, "!db.pg_float8");
    requireContains(printed, "!db.pg_int2");
    requireContains(printed, "!db.pg_int8");

    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_string_primitive_types_remain_distinct) {
    Fixture f;

    mlir::Type text = mlir::db::PgTextType::get(&f.ctx, kDefaultCollation);
    mlir::Type varchar = mlir::db::PgVarcharType::get(&f.ctx, kVarcharTypmod, kDefaultCollation);
    mlir::Type bpchar = mlir::db::PgBpcharType::get(&f.ctx, kBpcharTypmod, kDefaultCollation);

    REQUIRE(mlir::isa<mlir::db::PgTextType>(text));
    REQUIRE(mlir::isa<mlir::db::PgVarcharType>(varchar));
    REQUIRE(mlir::isa<mlir::db::PgBpcharType>(bpchar));
    REQUIRE(text != varchar);
    REQUIRE(text != bpchar);
    REQUIRE(varchar != bpchar);

    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(text), TEXTOID);
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(varchar), VARCHAROID);
    REQUIRE_EQ_U32(mlir::db::getPgTypeOid(bpchar), BPCHAROID);
    requireEqString(typeToString(mlir::db::getPgPhysicalCarrierType(text)), "!util.varlen32");
    requireEqString(typeToString(mlir::db::getPgPhysicalCarrierType(varchar)), "!util.varlen32");
    requireEqString(typeToString(mlir::db::getPgPhysicalCarrierType(bpchar)), "!util.varlen32");

    PG_RETURN_VOID();
}
