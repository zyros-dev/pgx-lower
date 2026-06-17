extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
}

#include "lingodb/runtime/helpers.h"
#include "pgx-lower/runtime/StringRuntime.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <string_view>
#include <vector>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

namespace {

auto makeVarLen32(std::vector<uint8_t>& bytes) -> ::runtime::VarLen32 {
    return ::runtime::VarLen32(bytes.data(), static_cast<uint32_t>(bytes.size()));
}

auto makeBytes(const std::string_view value) -> std::vector<uint8_t> {
    return std::vector<uint8_t>(value.begin(), value.end());
}

auto makePgStringDatum(const std::string_view value) -> Datum {
    return PointerGetDatum(cstring_to_text_with_len(value.data(), static_cast<int>(value.size())));
}

} // namespace

PGX_TEST_FN(string_runtime_bridge_bpchar_eq_matches_postgres) {
    auto leftBytes = makeBytes("ab ");
    auto rightBytes = makeBytes("ab   ");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_BPCHAREQ, DEFAULT_COLLATION_OID, makePgStringDatum("ab "), makePgStringDatum("ab   ")));
    const bool actual =
        ::runtime::StringRuntime::pgCallBool2(left, BPCHAROID, right, BPCHAROID, F_BPCHAREQ, DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_runtime_bridge_text_like_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto patternBytes = makeBytes("a%");
    auto value = makeVarLen32(valueBytes);
    auto pattern = makeVarLen32(patternBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXTLIKE, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("a%")));
    const bool actual =
        ::runtime::StringRuntime::pgCallBool2(value, TEXTOID, pattern, TEXTOID, F_TEXTLIKE, DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_runtime_bridge_text_not_like_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto patternBytes = makeBytes("b%");
    auto value = makeVarLen32(valueBytes);
    auto pattern = makeVarLen32(patternBytes);

    const bool expected = !DatumGetBool(
        OidFunctionCall2Coll(F_TEXTLIKE, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("b%")));
    const bool actual =
        !::runtime::StringRuntime::pgCallBool2(value, TEXTOID, pattern, TEXTOID, F_TEXTLIKE, DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}
