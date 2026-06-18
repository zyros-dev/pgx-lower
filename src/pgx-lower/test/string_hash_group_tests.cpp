extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/typcache.h"
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

auto makeBytes(const std::string_view value) -> std::vector<uint8_t> {
    return std::vector<uint8_t>(value.begin(), value.end());
}

auto makeVarLen32(std::vector<uint8_t>& bytes) -> ::runtime::VarLen32 {
    return ::runtime::VarLen32(bytes.data(), static_cast<uint32_t>(bytes.size()));
}

auto makePgStringDatum(const std::string_view value) -> Datum {
    return PointerGetDatum(cstring_to_text_with_len(value.data(), static_cast<int>(value.size())));
}

auto typeCacheStringHash(const Oid typeOid, const std::string_view value) -> uint64_t {
    TypeCacheEntry* typentry = lookup_type_cache(typeOid, TYPECACHE_EQ_OPR_FINFO | TYPECACHE_HASH_PROC_FINFO);
    REQUIRE(OidIsValid(typentry->eq_opr_finfo.fn_oid));
    REQUIRE(OidIsValid(typentry->hash_proc_finfo.fn_oid));
    return static_cast<uint64_t>(
        DatumGetUInt32(FunctionCall1Coll(&typentry->hash_proc_finfo, DEFAULT_COLLATION_OID, makePgStringDatum(value))));
}

auto bridgeStringHash(const Oid typeOid, const std::string_view value, const int32_t typmod = -1) -> uint64_t {
    auto bytes = makeBytes(value);
    auto varLen = makeVarLen32(bytes);
    return ::runtime::StringRuntime::pgHashString(varLen, typeOid, typmod, DEFAULT_COLLATION_OID);
}

} // namespace

PGX_TEST_FN(string_hash_text_uses_postgres_type_cache_hash) {
    REQUIRE(bridgeStringHash(TEXTOID, "alpha") == typeCacheStringHash(TEXTOID, "alpha"));
    REQUIRE(bridgeStringHash(TEXTOID, "\xc3\xa9"
                                      "clair")
            == typeCacheStringHash(TEXTOID, "\xc3\xa9"
                                            "clair"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_hash_varchar_uses_postgres_type_cache_hash) {
    REQUIRE(bridgeStringHash(VARCHAROID, "alpha", 16) == typeCacheStringHash(VARCHAROID, "alpha"));
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_hash_bpchar_uses_postgres_type_cache_hash) {
    REQUIRE(bridgeStringHash(BPCHAROID, "ab   ", 8) == typeCacheStringHash(BPCHAROID, "ab   "));
    PG_RETURN_VOID();
}

PGX_TEST_FN(string_hash_bpchar_obeys_postgres_equality_hash_contract) {
    auto leftBytes = makeBytes("ab ");
    auto rightBytes = makeBytes("ab   ");
    auto left = makeVarLen32(leftBytes);
    auto right = makeVarLen32(rightBytes);

    const bool equal = ::runtime::StringRuntime::pgCallBool2(left, BPCHAROID, right, BPCHAROID, F_BPCHAREQ,
                                                             DEFAULT_COLLATION_OID);
    const uint64_t leftHash = ::runtime::StringRuntime::pgHashString(left, BPCHAROID, 8, DEFAULT_COLLATION_OID);
    const uint64_t rightHash = ::runtime::StringRuntime::pgHashString(right, BPCHAROID, 8, DEFAULT_COLLATION_OID);

    REQUIRE(equal);
    REQUIRE(leftHash == rightHash);
    PG_RETURN_VOID();
}
