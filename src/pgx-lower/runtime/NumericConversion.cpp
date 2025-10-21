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

typedef int16 NumericDigit;

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
    int32 vl_len_;
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

__int128 numeric_to_i128(Datum numeric_datum, int32_t target_scale) {
    PGX_IO(RUNTIME);

    const Numeric num = DatumGetNumeric(numeric_datum);

    // Handle special values
    if (NUMERIC_IS_NAN(num) || NUMERIC_IS_INF(num)) {
        PGX_LOG(RUNTIME, WARNING_LEVEL, "numeric_to_i128: NaN or Inf encountered, returning 0");
        return 0;
    }

    const bool is_negative = (NUMERIC_SIGN(num) == NUMERIC_NEG);
    const int weight = NUMERIC_WEIGHT(num);
    const int dscale = NUMERIC_DSCALE(num);
    const int ndigits = NUMERIC_NDIGITS(num);
    const NumericDigit* digits = NUMERIC_DIGITS(num);

    PGX_LOG(RUNTIME, TRACE, "numeric_to_i128: weight=%d, dscale=%d, ndigits=%d, target_scale=%d", weight, dscale,
            ndigits, target_scale);

    if (ndigits == 0) {
        return 0;
    }

    __int128 value = 0;
    for (int i = 0; i < ndigits; i++) {
        value = value * NBASE + digits[i];
    }

    const int total_base_digits = ndigits;
    const int base_digits_before_decimal = weight + 1;
    const int base_digits_after_decimal = total_base_digits - base_digits_before_decimal;

    const int current_decimal_scale = base_digits_after_decimal * DEC_DIGITS;
    const int scale_adjustment = target_scale - current_decimal_scale;

    if (scale_adjustment > 0) {
        for (int i = 0; i < scale_adjustment; i++) {
            value *= 10;
        }
    } else if (scale_adjustment < 0) {
        for (int i = 0; i < -scale_adjustment; i++) {
            value /= 10;
        }
    }

    return is_negative ? -value : value;
}

Datum i128_to_numeric(__int128 value, int32_t scale) {
    PGX_IO(RUNTIME);

    PGX_LOG(RUNTIME, TRACE, "i128_to_numeric: value=%lld, scale=%d", static_cast<long long>(value), scale);

    char value_str[45];
    const bool is_negative = (value < 0);
    __uint128_t abs_value = is_negative ? -static_cast<__uint128_t>(value) : static_cast<__uint128_t>(value);

    char* p = value_str + sizeof(value_str) - 1;
    *p = '\0';

    do {
        *--p = '0' + (abs_value % 10);
        abs_value /= 10;
    } while (abs_value > 0 && p > value_str);

    if (is_negative && p > value_str) {
        *--p = '-';
    }

    size_t len = strlen(p);
    char* end = p + len - 1;
    int zeros_removed = 0;
    while (end > p && *end == '0' && zeros_removed < scale) {
        *end-- = '\0';
        zeros_removed++;
    }
    scale -= zeros_removed;
    len = strlen(p);

    char buffer[128];
    if (scale > 0) {
        if (len <= static_cast<size_t>(scale)) {
            const int leading_zeros = scale - len + 1;
            buffer[0] = '0';
            buffer[1] = '.';
            int pos = 2;
            for (int i = 1; i < leading_zeros; i++) {
                buffer[pos++] = '0';
            }
            strcpy(buffer + pos, p);
        } else {
            // Number >= 1, insert decimal point
            const size_t integer_len = len - scale;
            strncpy(buffer, p, integer_len);
            buffer[integer_len] = '.';
            strcpy(buffer + integer_len + 1, p + integer_len);
        }
    } else {
        // No decimal point needed
        strcpy(buffer, p);
    }

    PGX_LOG(RUNTIME, TRACE, "i128_to_numeric: formatted='%s'", buffer);

    const Datum numeric_datum = DirectFunctionCall3(numeric_in, CStringGetDatum(buffer), ObjectIdGetDatum(InvalidOid),
                                                    Int32GetDatum(-1));

    const Datum copied_datum = datumCopy(numeric_datum, false, -1);

    return copied_datum;
}
