#include "postgres.h"
#include "fmgr.h"
#include "varatt.h"

#include <string.h>

#include "pgx-lower/runtime/NumericConversion.h"

#define NBASE 10000
#define NUMERIC_POS 0x0000
#define NUMERIC_NEG 0x4000
#define NUMERIC_DSCALE_MASK 0x3FFF

typedef int16 NumericDigit;

struct test_numeric_long {
    uint16 n_sign_dscale;
    int16  n_weight;
    NumericDigit n_data[1];
};

union test_numeric_choice {
    uint16 n_header;
    struct test_numeric_long n_long;
};

struct test_numeric_data {
    int32 vl_len_;
    union test_numeric_choice choice;
};

#define TEST_NUMERIC_HDR (VARHDRSZ + sizeof(uint16) + sizeof(int16))

static Datum
build_numeric(bool neg, int16 weight, uint16 dscale,
              const NumericDigit *digits, int ndigits)
{
    Size total = TEST_NUMERIC_HDR + (Size) ndigits * sizeof(NumericDigit);
    struct test_numeric_data *num = (struct test_numeric_data *) palloc0(total);
    SET_VARSIZE(num, total);
    num->choice.n_long.n_sign_dscale =
        (neg ? NUMERIC_NEG : NUMERIC_POS) | (dscale & NUMERIC_DSCALE_MASK);
    num->choice.n_long.n_weight = weight;
    if (ndigits > 0)
        memcpy(num->choice.n_long.n_data, digits, (size_t) ndigits * sizeof(NumericDigit));
    return PointerGetDatum(num);
}

#define ASSERT_EQ_I128(actual, expected) \
    do { \
        __int128 _a = (actual); \
        __int128 _e = (expected); \
        if (_a != _e) \
            elog(ERROR, "%s:%d expected %lld got %lld", \
                 __FILE__, __LINE__, (long long) _e, (long long) _a); \
    } while (0)

PG_FUNCTION_INFO_V1(ts_test_numeric_to_i128_zero);
Datum
ts_test_numeric_to_i128_zero(PG_FUNCTION_ARGS)
{
    Datum d = build_numeric(false, 0, 0, NULL, 0);
    ASSERT_EQ_I128(numeric_to_i128(d, 0), 0);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_numeric_to_i128_positive);
Datum
ts_test_numeric_to_i128_positive(PG_FUNCTION_ARGS)
{
    NumericDigit digits[] = {1, 2345};
    Datum d = build_numeric(false, 1, 0, digits, 2);
    ASSERT_EQ_I128(numeric_to_i128(d, 0), 12345);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_numeric_to_i128_negative);
Datum
ts_test_numeric_to_i128_negative(PG_FUNCTION_ARGS)
{
    NumericDigit digits[] = {1, 2345};
    Datum d = build_numeric(true, 1, 0, digits, 2);
    ASSERT_EQ_I128(numeric_to_i128(d, 0), -12345);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_numeric_to_i128_rescale_up);
Datum
ts_test_numeric_to_i128_rescale_up(PG_FUNCTION_ARGS)
{
    NumericDigit digits[] = {1, 5000};
    Datum d = build_numeric(false, 0, 1, digits, 2);
    ASSERT_EQ_I128(numeric_to_i128(d, 4), 15000);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_numeric_to_i128_scale_down);
Datum
ts_test_numeric_to_i128_scale_down(PG_FUNCTION_ARGS)
{
    NumericDigit digits[] = {1, 2345};
    Datum d = build_numeric(false, 1, 0, digits, 2);
    ASSERT_EQ_I128(numeric_to_i128(d, -2), 123);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(ts_test_numeric_to_i128_large);
Datum
ts_test_numeric_to_i128_large(PG_FUNCTION_ARGS)
{
    NumericDigit digits[] = {9999, 9999};
    Datum d = build_numeric(false, 1, 0, digits, 2);
    ASSERT_EQ_I128(numeric_to_i128(d, 0), 99999999LL);
    PG_RETURN_VOID();
}
