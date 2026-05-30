#include "pgx-lower/runtime/NumericConversion.h"
#include "pgx-lower/utility/logging.h"

#include <varatt.h>

extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/numeric.h"
#include "utils/datum.h"
}

#include <cstring>
#include <cctype>

#define NBASE 10000
#define DEC_DIGITS 4

using NumericDigit = int16;

struct NumericShort {
    uint16 n_header;
    NumericDigit n_data[FLEXIBLE_ARRAY_MEMBER];
};

struct NumericLong {
    uint16 n_sign_dscale;
    int16 n_weight;
    NumericDigit n_data[FLEXIBLE_ARRAY_MEMBER];
};

union NumericChoice {
    uint16 n_header;
    struct NumericLong n_long;
    struct NumericShort n_short;
};

struct NumericData {
    int32 vl_len;
    union NumericChoice choice;
};

// Flag bits
#define NUMERIC_SIGN_MASK 0xC000
#define NUMERIC_POS 0x0000
#define NUMERIC_NEG 0x4000
#define NUMERIC_SHORT 0x8000
#define NUMERIC_SPECIAL 0xC000

#define NUMERIC_FLAGBITS(n) ((n)->choice.n_header & NUMERIC_SIGN_MASK)
#define NUMERIC_IS_SHORT(n) (NUMERIC_FLAGBITS(n) == NUMERIC_SHORT)
#define NUMERIC_IS_SPECIAL(n) (NUMERIC_FLAGBITS(n) == NUMERIC_SPECIAL)

// Special values
#define NUMERIC_EXT_SIGN_MASK 0xF000
#define NUMERIC_NAN 0xC000
#define NUMERIC_PINF 0xD000
#define NUMERIC_NINF 0xF000
#define NUMERIC_IS_NAN(n) ((n)->choice.n_header == NUMERIC_NAN)
#define NUMERIC_IS_INF(n) (((n)->choice.n_header & ~0x2000) == NUMERIC_PINF)

// Short format
#define NUMERIC_SHORT_SIGN_MASK 0x2000
#define NUMERIC_SHORT_DSCALE_MASK 0x1F80
#define NUMERIC_SHORT_DSCALE_SHIFT 7
#define NUMERIC_SHORT_WEIGHT_SIGN_MASK 0x0040
#define NUMERIC_SHORT_WEIGHT_MASK 0x003F

// Extract sign
#define NUMERIC_SIGN(n)                                                                                                \
    (NUMERIC_IS_SHORT(n)                                                                                               \
         ? (((n)->choice.n_short.n_header & NUMERIC_SHORT_SIGN_MASK) ? NUMERIC_NEG : NUMERIC_POS)                      \
         : (NUMERIC_IS_SPECIAL(n) ? ((n)->choice.n_header & NUMERIC_EXT_SIGN_MASK) : NUMERIC_FLAGBITS(n)))

// Extract display scale
#define NUMERIC_DSCALE_MASK 0x3FFF
#define NUMERIC_HEADER_IS_SHORT(n) (((n)->choice.n_header & 0x8000) != 0)
#define NUMERIC_DSCALE(n)                                                                                              \
    (NUMERIC_HEADER_IS_SHORT((n))                                                                                      \
         ? ((n)->choice.n_short.n_header & NUMERIC_SHORT_DSCALE_MASK) >> NUMERIC_SHORT_DSCALE_SHIFT                    \
         : ((n)->choice.n_long.n_sign_dscale & NUMERIC_DSCALE_MASK))

// Extract weight
#define NUMERIC_WEIGHT(n)                                                                                              \
    (NUMERIC_HEADER_IS_SHORT((n))                                                                                      \
         ? (((n)->choice.n_short.n_header & NUMERIC_SHORT_WEIGHT_SIGN_MASK ? ~NUMERIC_SHORT_WEIGHT_MASK : 0)           \
            | ((n)->choice.n_short.n_header & NUMERIC_SHORT_WEIGHT_MASK))                                              \
         : ((n)->choice.n_long.n_weight))

// Access digits array
#define NUMERIC_HEADER_SIZE(n) (VARHDRSZ + sizeof(uint16) + (NUMERIC_HEADER_IS_SHORT(n) ? 0 : sizeof(int16)))
#define NUMERIC_DIGITS(num) (NUMERIC_HEADER_IS_SHORT(num) ? (num)->choice.n_short.n_data : (num)->choice.n_long.n_data)
#define NUMERIC_NDIGITS(num) ((VARSIZE(num) - NUMERIC_HEADER_SIZE(num)) / sizeof(NumericDigit))

// Size of numeric header (long format)
#define NUMERIC_HDRSZ (VARHDRSZ + sizeof(uint16) + sizeof(int16))

__int128 numeric_to_i128(Datum numeric_datum, int32_t target_scale) {
    PGX_IO(RUNTIME);

    const Numeric NUM = DatumGetNumeric(numeric_datum);

    // Handle special values
    if (NUMERIC_IS_NAN(NUM) || NUMERIC_IS_INF(NUM)) {
        PGX_LOG(RUNTIME, WARNING_LEVEL, "numeric_to_i128: NaN or Inf encountered, returning 0");
        return 0;
    }

    const bool IS_NEGATIVE = (NUMERIC_SIGN(NUM) == NUMERIC_NEG);
    const int WEIGHT = NUMERIC_WEIGHT(NUM);
    const int DSCALE = NUMERIC_DSCALE(NUM);
    const int NDIGITS = NUMERIC_NDIGITS(NUM);
    const NumericDigit* const digits = NUMERIC_DIGITS(NUM);

    PGX_LOG(RUNTIME, TRACE, "numeric_to_i128: weight=%d, dscale=%d, ndigits=%d, target_scale=%d", WEIGHT, DSCALE,
            NDIGITS, target_scale);

    if (NDIGITS == 0) {
        return 0;
    }

    __int128 value = 0;
    for (int i = 0; i < NDIGITS; i++) {
        value = value * NBASE + digits[i];
    }

    const int TOTAL_BASE_DIGITS = NDIGITS;
    const int BASE_DIGITS_BEFORE_DECIMAL = WEIGHT + 1;
    const int BASE_DIGITS_AFTER_DECIMAL = TOTAL_BASE_DIGITS - BASE_DIGITS_BEFORE_DECIMAL;

    const int CURRENT_DECIMAL_SCALE = BASE_DIGITS_AFTER_DECIMAL * DEC_DIGITS;
    const int SCALE_ADJUSTMENT = target_scale - CURRENT_DECIMAL_SCALE;

    if (SCALE_ADJUSTMENT > 0) {
        for (int i = 0; i < SCALE_ADJUSTMENT; i++) {
            value *= 10;
        }
    } else if (SCALE_ADJUSTMENT < 0) {
        for (int i = 0; i < -SCALE_ADJUSTMENT; i++) {
            value /= 10;
        }
    }

    return IS_NEGATIVE ? -value : value;
}

Datum i128_to_numeric(__int128 value, int32_t scale) {
    PGX_IO(RUNTIME);

    PGX_LOG(RUNTIME, TRACE, "i128_to_numeric: value=%lld, scale=%d", static_cast<long long>(value), scale);

    if (value == 0) {
        return DirectFunctionCall1(int4_numeric, Int32GetDatum(0));
    }

    const bool IS_NEGATIVE = (value < 0);
    __uint128_t const abs_value = IS_NEGATIVE ? -static_cast<__uint128_t>(value) : static_cast<__uint128_t>(value);

    __uint128_t scale_divisor = 1;
    for (int i = 0; i < scale; i++) {
        scale_divisor *= 10;
    }

    __uint128_t const integer_part = abs_value / scale_divisor;
    __uint128_t fractional_part = abs_value % scale_divisor;

    int actual_frac_scale = scale;
    while (actual_frac_scale > 0 && fractional_part % 10 == 0) {
        fractional_part /= 10;
        actual_frac_scale--;
    }

    const int FRAC_PADDING = (DEC_DIGITS - (actual_frac_scale % DEC_DIGITS)) % DEC_DIGITS;
    for (int i = 0; i < FRAC_PADDING; i++) {
        fractional_part *= 10;
    }

    NumericDigit int_digits[40];
    int int_ndigits = 0;
    __uint128_t temp = integer_part;
    if (temp == 0) {
        int_digits[int_ndigits++] = 0;
    } else {
        while (temp > 0) {
            int_digits[int_ndigits++] = temp % NBASE;
            temp /= NBASE;
        }
    }

    NumericDigit frac_digits[40];
    int frac_ndigits = 0;
    temp = fractional_part;
    const int EXPECTED_FRAC_DIGITS = (actual_frac_scale + FRAC_PADDING) / DEC_DIGITS;
    while (frac_ndigits < EXPECTED_FRAC_DIGITS) {
        frac_digits[frac_ndigits++] = temp % NBASE;
        temp /= NBASE;
    }

    NumericDigit digits[40];
    int ndigits = 0;

    for (int i = int_ndigits - 1; i >= 0; i--) {
        digits[ndigits++] = int_digits[i];
    }

    for (int i = frac_ndigits - 1; i >= 0; i--) {
        digits[ndigits++] = frac_digits[i];
    }

    while (ndigits > int_ndigits && digits[ndigits - 1] == 0) {
        ndigits--;
    }

    const int WEIGHT = int_ndigits - 1;

    const int NUMERIC_SIZE = NUMERIC_HDRSZ + ndigits * sizeof(NumericDigit);
    Numeric const result = static_cast<Numeric>(palloc(NUMERIC_SIZE));

    SET_VARSIZE(result, NUMERIC_SIZE);

    result->choice.n_header = IS_NEGATIVE ? NUMERIC_NEG : NUMERIC_POS;
    result->choice.n_long.n_sign_dscale = (IS_NEGATIVE ? NUMERIC_NEG : NUMERIC_POS) | (actual_frac_scale & NUMERIC_DSCALE_MASK);
    result->choice.n_long.n_weight = WEIGHT;

    memcpy(result->choice.n_long.n_data, digits, ndigits * sizeof(NumericDigit));

    return NumericGetDatum(result);
}
