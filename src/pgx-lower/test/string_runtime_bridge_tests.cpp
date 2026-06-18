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

#include <cstring>
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

auto datumPayloadMatchesVarLen32(Datum datum, ::runtime::VarLen32 value) -> bool {
    const auto* varlena = reinterpret_cast<const struct varlena*>(DatumGetPointer(datum));
    return VARSIZE_ANY_EXHDR(varlena) == value.getLen()
           && std::memcmp(VARDATA_ANY(varlena), value.data(), value.getLen()) == 0;
}

} // namespace

PGX_TEST_FN(string_bridge_text_eq_matches_postgres) {
    auto leftBytes = makeBytes("alpha");
    auto rightBytes = makeBytes("alpha");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXTEQ, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("alpha")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(left, TEXTOID, right, TEXTOID, F_TEXTEQ,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_text_lt_uses_collation) {
    auto leftBytes = makeBytes("alpha");
    auto rightBytes = makeBytes("beta");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXT_LT, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("beta")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(left, TEXTOID, right, TEXTOID, F_TEXT_LT,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_varchar_eq_matches_postgres) {
    auto leftBytes = makeBytes("alpha");
    auto rightBytes = makeBytes("alpha");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXTEQ, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("alpha")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(left, VARCHAROID, right, VARCHAROID, F_TEXTEQ,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_varchar_lt_matches_postgres) {
    auto leftBytes = makeBytes("alpha");
    auto rightBytes = makeBytes("beta");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXT_LT, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("beta")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(left, VARCHAROID, right, VARCHAROID, F_TEXT_LT,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_runtime_bridge_bpchar_eq_matches_postgres) {
    auto leftBytes = makeBytes("ab ");
    auto rightBytes = makeBytes("ab   ");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_BPCHAREQ, DEFAULT_COLLATION_OID, makePgStringDatum("ab "), makePgStringDatum("ab   ")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(left, BPCHAROID, right, BPCHAROID, F_BPCHAREQ,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_bpchar_lt_matches_postgres) {
    auto leftBytes = makeBytes("ab ");
    auto rightBytes = makeBytes("ac ");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_BPCHARLT, DEFAULT_COLLATION_OID, makePgStringDatum("ab "), makePgStringDatum("ac ")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(left, BPCHAROID, right, BPCHAROID, F_BPCHARLT,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_runtime_bridge_bpchar_hash_matches_postgres) {
    auto valueBytes = makeBytes("ab   ");
    auto value = makeVarLen32(valueBytes);

    const uint64_t expected = static_cast<uint64_t>(
        DatumGetUInt32(OidFunctionCall1Coll(F_HASHBPCHAR, DEFAULT_COLLATION_OID, makePgStringDatum("ab   "))));
    const uint64_t actual = ::runtime::StringRuntime::pgCallHash1(value, BPCHAROID, F_HASHBPCHAR, DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_runtime_bridge_bpchar_hash_agrees_with_equality_across_padding) {
    auto leftBytes = makeBytes("ab ");
    auto rightBytes = makeBytes("ab   ");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool equal = ::runtime::StringRuntime::pgCallBool2(left, BPCHAROID, right, BPCHAROID, F_BPCHAREQ,
                                                             DEFAULT_COLLATION_OID);
    const uint64_t leftHash = ::runtime::StringRuntime::pgCallHash1(left, BPCHAROID, F_HASHBPCHAR, DEFAULT_COLLATION_OID);
    const uint64_t rightHash = ::runtime::StringRuntime::pgCallHash1(right, BPCHAROID, F_HASHBPCHAR,
                                                                     DEFAULT_COLLATION_OID);

    REQUIRE(equal);
    REQUIRE(leftHash == rightHash);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_runtime_bridge_text_like_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto patternBytes = makeBytes("a%");
    auto value = makeVarLen32(valueBytes);
    auto pattern = makeVarLen32(patternBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXTLIKE, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("a%")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(value, TEXTOID, pattern, TEXTOID, F_TEXTLIKE,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_varchar_like_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto patternBytes = makeBytes("a%");
    auto value = makeVarLen32(valueBytes);
    auto pattern = makeVarLen32(patternBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_TEXTLIKE, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("a%")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(value, VARCHAROID, pattern, TEXTOID, F_TEXTLIKE,
                                                              DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_bpchar_like_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto patternBytes = makeBytes("a%");
    auto value = makeVarLen32(valueBytes);
    auto pattern = makeVarLen32(patternBytes);

    const bool expected = DatumGetBool(
        OidFunctionCall2Coll(F_BPCHARLIKE, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"), makePgStringDatum("a%")));
    const bool actual = ::runtime::StringRuntime::pgCallBool2(value, BPCHAROID, pattern, TEXTOID, F_BPCHARLIKE,
                                                              DEFAULT_COLLATION_OID);

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
    const bool actual = !::runtime::StringRuntime::pgCallBool2(value, TEXTOID, pattern, TEXTOID, F_TEXTLIKE,
                                                               DEFAULT_COLLATION_OID);

    REQUIRE(actual == expected);
    REQUIRE(actual);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_substring_is_character_based) {
    auto valueBytes = makeBytes("\xc3\xa9"
                                "clair");
    auto value = makeVarLen32(valueBytes);

    const Datum expected = OidFunctionCall3Coll(F_SUBSTRING_TEXT_INT4_INT4, DEFAULT_COLLATION_OID,
                                                makePgStringDatum("\xc3\xa9"
                                                                  "clair"),
                                                Int32GetDatum(1), Int32GetDatum(1));
    auto actual = ::runtime::StringRuntime::pgCallString3(value, TEXTOID, 1, 1, F_SUBSTRING_TEXT_INT4_INT4,
                                                          DEFAULT_COLLATION_OID);

    REQUIRE(datumPayloadMatchesVarLen32(expected, actual));
    REQUIRE(actual.getLen() == 2);
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_varchar_substring_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto value = makeVarLen32(valueBytes);

    const Datum expected = OidFunctionCall3Coll(F_SUBSTRING_TEXT_INT4_INT4, DEFAULT_COLLATION_OID,
                                                makePgStringDatum("alpha"), Int32GetDatum(2), Int32GetDatum(3));
    auto actual = ::runtime::StringRuntime::pgCallString3(value, VARCHAROID, 2, 3, F_SUBSTRING_TEXT_INT4_INT4,
                                                          DEFAULT_COLLATION_OID);

    REQUIRE(datumPayloadMatchesVarLen32(expected, actual));
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_substring_without_length_matches_postgres) {
    auto valueBytes = makeBytes("alpha");
    auto value = makeVarLen32(valueBytes);

    const Datum expected = OidFunctionCall2Coll(F_SUBSTRING_TEXT_INT4, DEFAULT_COLLATION_OID,
                                                makePgStringDatum("alpha"), Int32GetDatum(2));
    const auto actual = ::runtime::StringRuntime::pgCallString2(value, TEXTOID, 2, F_SUBSTRING_TEXT_INT4,
                                                                DEFAULT_COLLATION_OID);

    REQUIRE(datumPayloadMatchesVarLen32(expected, actual));
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_upper_lower_match_postgres) {
    auto upperBytes = makeBytes("alpha");
    auto lowerBytes = makeBytes("BETA");
    auto upperValue = makeVarLen32(upperBytes);
    auto lowerValue = makeVarLen32(lowerBytes);

    const Datum expectedUpper = OidFunctionCall1Coll(F_UPPER_TEXT, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"));
    const Datum expectedLower = OidFunctionCall1Coll(F_LOWER_TEXT, DEFAULT_COLLATION_OID, makePgStringDatum("BETA"));
    const auto actualUpper = ::runtime::StringRuntime::pgCallString1(upperValue, TEXTOID, F_UPPER_TEXT,
                                                                     DEFAULT_COLLATION_OID);
    const auto actualLower = ::runtime::StringRuntime::pgCallString1(lowerValue, TEXTOID, F_LOWER_TEXT,
                                                                     DEFAULT_COLLATION_OID);

    REQUIRE(datumPayloadMatchesVarLen32(expectedUpper, actualUpper));
    REQUIRE(datumPayloadMatchesVarLen32(expectedLower, actualLower));
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_bridge_varchar_upper_lower_match_postgres) {
    auto upperBytes = makeBytes("alpha");
    auto lowerBytes = makeBytes("BETA");
    auto upperValue = makeVarLen32(upperBytes);
    auto lowerValue = makeVarLen32(lowerBytes);

    const Datum expectedUpper = OidFunctionCall1Coll(F_UPPER_TEXT, DEFAULT_COLLATION_OID, makePgStringDatum("alpha"));
    const Datum expectedLower = OidFunctionCall1Coll(F_LOWER_TEXT, DEFAULT_COLLATION_OID, makePgStringDatum("BETA"));
    const auto actualUpper = ::runtime::StringRuntime::pgCallString1(upperValue, VARCHAROID, F_UPPER_TEXT,
                                                                     DEFAULT_COLLATION_OID);
    const auto actualLower = ::runtime::StringRuntime::pgCallString1(lowerValue, VARCHAROID, F_LOWER_TEXT,
                                                                     DEFAULT_COLLATION_OID);

    REQUIRE(datumPayloadMatchesVarLen32(expectedUpper, actualUpper));
    REQUIRE(datumPayloadMatchesVarLen32(expectedLower, actualLower));
    PG_RETURN_VOID();
}
