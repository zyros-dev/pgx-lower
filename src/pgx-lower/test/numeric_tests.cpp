extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "varatt.h"
#include "utils/numeric.h"
#include "utils/fmgrprotos.h"
}

#include <cstring>
#include <vector>

#include "pgx-lower/runtime/NumericConversion.h"
#include "pgx-lower/runtime/NumericRuntime.h"
#include "pgx-lower/test/pgx_test_fn.h"

namespace {

#define NBASE 10000
#define NUMERIC_POS 0x0000
#define NUMERIC_NEG 0x4000
#define NUMERIC_DSCALE_MASK 0x3FFF

typedef int16 NumericDigit;

struct NumericLong {
    uint16 n_sign_dscale;
    int16 n_weight;
    NumericDigit n_data[1];
};

union NumericChoice {
    uint16 n_header;
    struct NumericLong n_long;
};

struct NumericData {
    int32 vl_len_;
    union NumericChoice choice;
};

Datum make_numeric(std::vector<char>& buf, bool neg, int16 weight, uint16 dscale,
                   const std::vector<NumericDigit>& digits) {
    const size_t hdr = VARHDRSZ + sizeof(uint16) + sizeof(int16);
    const size_t total = hdr + digits.size() * sizeof(NumericDigit);
    buf.assign(total, 0);
    auto* num = reinterpret_cast<NumericData*>(buf.data());
    SET_VARSIZE(num, total);
    num->choice.n_long.n_sign_dscale = (neg ? NUMERIC_NEG : NUMERIC_POS) | (dscale & NUMERIC_DSCALE_MASK);
    num->choice.n_long.n_weight = weight;
    std::memcpy(num->choice.n_long.n_data, digits.data(), digits.size() * sizeof(NumericDigit));
    return PointerGetDatum(num);
}

}  // namespace

#define ASSERT_EQ_I128(actual, expected) \
    do { \
        __int128 _a = (actual); \
        __int128 _e = (expected); \
        if (_a != _e) \
            elog(ERROR, "%s:%d expected %lld got %lld", \
                 __FILE__, __LINE__, (long long) _e, (long long) _a); \
    } while (0)

#define ASSERT_NUMERIC_EQ_STR(actual_datum, expected_cstr) \
    do { \
        char* _s = DatumGetCString(DirectFunctionCall1(numeric_out, (actual_datum))); \
        if (std::strcmp(_s, (expected_cstr)) != 0) \
            elog(ERROR, "%s:%d expected numeric '%s' got '%s'", \
                 __FILE__, __LINE__, (expected_cstr), _s); \
    } while (0)

PGX_TEST_FN(numeric_add_basic) {
    std::vector<char> buf_l;
    std::vector<char> buf_r;
    Datum l = make_numeric(buf_l, false, 1, 0, {1, 2345}); // 12345
    Datum r = make_numeric(buf_r, false, 0, 0, {6789});    // 6789
    ASSERT_NUMERIC_EQ_STR(pgx_numeric_add(l, r), "19134");
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_to_i128_zero) {
    std::vector<char> buf;
    ASSERT_EQ_I128(numeric_to_i128(make_numeric(buf, false, 0, 0, {}), 0), 0);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_to_i128_positive) {
    std::vector<char> buf;
    ASSERT_EQ_I128(numeric_to_i128(make_numeric(buf, false, 1, 0, {1, 2345}), 0), 12345);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_to_i128_negative) {
    std::vector<char> buf;
    ASSERT_EQ_I128(numeric_to_i128(make_numeric(buf, true, 1, 0, {1, 2345}), 0), -12345);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_to_i128_rescale_up) {
    std::vector<char> buf;
    ASSERT_EQ_I128(numeric_to_i128(make_numeric(buf, false, 0, 1, {1, 5000}), 4), 15000);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_to_i128_scale_down) {
    std::vector<char> buf;
    ASSERT_EQ_I128(numeric_to_i128(make_numeric(buf, false, 1, 0, {1, 2345}), -2), 123);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_to_i128_large) {
    std::vector<char> buf;
    ASSERT_EQ_I128(numeric_to_i128(make_numeric(buf, false, 1, 0, {9999, 9999}), 0), 99999999LL);
    PG_RETURN_VOID();
}
