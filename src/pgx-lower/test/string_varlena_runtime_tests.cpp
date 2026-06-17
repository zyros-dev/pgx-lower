extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
}

#include "lingodb/runtime/helpers.h"
#include "pgx-lower/runtime/PostgreSQLRuntime.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/runtime_templates.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <cstring>
#include <array>
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
        auto _a = static_cast<uint32_t>(actual);                                                                       \
        auto _e = static_cast<uint32_t>(expected);                                                                     \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

namespace {

auto makeVarLen32(std::vector<uint8_t>& bytes) -> ::runtime::VarLen32 {
    return ::runtime::VarLen32(bytes.data(), static_cast<uint32_t>(bytes.size()));
}

auto makeAsciiBytes(const std::string& value) -> std::vector<uint8_t> {
    return std::vector<uint8_t>(value.begin(), value.end());
}

auto makeLazyVarLen32(const std::string& value) -> ::runtime::VarLen32 {
    static_assert(sizeof(::runtime::VarLen32) == 16, "VarLen32 layout changed");

    std::array<uint8_t, sizeof(::runtime::VarLen32)> raw{};
    const uint32_t len_with_flag = static_cast<uint32_t>(value.size()) | ::runtime::VarLen32::lazyMask;
    std::memcpy(raw.data(), &len_with_flag, sizeof(len_with_flag));

    auto* ptr = const_cast<char*>(value.data());
    std::memcpy(raw.data() + 8, &ptr, sizeof(ptr));

    uint8_t dummy = 0;
    ::runtime::VarLen32 varlen(&dummy, 0);
    std::memcpy(&varlen, raw.data(), raw.size());
    return varlen;
}

void resetComputedResults() {
    g_computed_results.clear();
    prepare_computed_results(1);
}

void requireStoredStringEquals(int columnIndex, const std::string& expected) {
    REQUIRE(columnIndex < g_computed_results.numComputedColumns);
    REQUIRE(!g_computed_results.computedNulls[columnIndex]);
    const auto datum = g_computed_results.computedValues[columnIndex];
    const auto* pgText = DatumGetTextPP(datum);
    REQUIRE(VARSIZE_ANY_EXHDR(pgText) == static_cast<int32>(expected.size()));
    REQUIRE(std::memcmp(VARDATA_ANY(pgText), expected.data(), expected.size()) == 0);
}

} // namespace

PGX_TEST_FN(string_varlena_runtime_varlen32_has_no_pg_identity) {
    REQUIRE_EQ_U32(pgx_lower::runtime::getTypeOid<::runtime::VarLen32>(), InvalidOid);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_varlena_empty_roundtrip) {
    resetComputedResults();
    g_computed_results.setMetadata(0, {TEXTOID, -1, DEFAULT_COLLATION_OID});

    std::vector<uint8_t> emptyBytes;
    ::runtime::TableBuilder builder;
    builder.addBinary(true, makeVarLen32(emptyBytes));

    REQUIRE(!g_computed_results.computedNulls[0]);
    requireStoredStringEquals(0, "");
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_varlena_short_header_roundtrip) {
    resetComputedResults();
    g_computed_results.setMetadata(0, {VARCHAROID, 16, DEFAULT_COLLATION_OID});

    auto bytes = makeAsciiBytes("short");
    ::runtime::TableBuilder builder;
    builder.addBinary(true, makeVarLen32(bytes));

    REQUIRE_EQ_U32(g_computed_results.computedMetadata[0].type_oid, VARCHAROID);
    requireStoredStringEquals(0, "short");
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_varlena_long_roundtrip) {
    resetComputedResults();
    g_computed_results.setMetadata(0, {BPCHAROID, 68, DEFAULT_COLLATION_OID});

    std::string longString(65536, 'x');
    auto bytes = makeAsciiBytes(longString);
    ::runtime::TableBuilder builder;
    builder.addBinary(true, makeVarLen32(bytes));

    REQUIRE_EQ_U32(g_computed_results.computedMetadata[0].type_oid, BPCHAROID);
    requireStoredStringEquals(0, longString);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_varlena_lazy_scan_layout_roundtrip) {
    resetComputedResults();
    g_computed_results.setMetadata(0, {TEXTOID, -1, DEFAULT_COLLATION_OID});

    const std::string value = "AF";
    ::runtime::VarLen32 lazyValue = makeLazyVarLen32(value);

    REQUIRE_EQ_U32(lazyValue.getLen(), static_cast<uint32_t>(value.size()));
    REQUIRE(std::memcmp(lazyValue.getPtr(), value.data(), value.size()) == 0);

    ::runtime::TableBuilder builder;
    builder.addBinary(true, lazyValue);

    requireStoredStringEquals(0, value);
    PG_RETURN_VOID();
}
