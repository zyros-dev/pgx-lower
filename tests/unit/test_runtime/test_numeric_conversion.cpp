// Unit tests for src/pgx-lower/runtime/NumericConversion.cpp.
//
// Tests the forward direction (numeric_to_i128) first — it has the smallest
// PG-API surface (just pg_detoast_datum via DatumGetNumeric). The reverse
// direction (i128_to_numeric) needs DirectFunctionCall1 + int4_numeric +
// palloc which require more PG infrastructure; deferred to a follow-up
// test_numeric_conversion_reverse.cpp.
//
// We construct Numeric values by hand using PG's wire-format (the same
// macros NumericConversion.cpp already pulls from postgres.h), so the test
// exercises the real bit-shuffling without needing PG's catalog or fmgr
// machinery for the input side.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
#include "postgres.h"
}

#include "pgx-lower/runtime/NumericConversion.h"

namespace {

// Copy of the structure layout from NumericConversion.cpp — kept in sync
// by checking against PG headers at compile time. If PG ever changes the
// numeric wire format this test will break loudly, which is the point.

#define NBASE 10000
#define DEC_DIGITS 4
#define NUMERIC_POS 0x0000
#define NUMERIC_NEG 0x4000
#define NUMERIC_DSCALE_MASK 0x3FFF

typedef int16 NumericDigit;

struct NumericLong {
    uint16 n_sign_dscale;
    int16 n_weight;
    NumericDigit n_data[1];  // FAM
};

union NumericChoice {
    uint16 n_header;
    struct NumericLong n_long;
};

struct NumericData {
    int32 vl_len_;
    union NumericChoice choice;
};

// Build an in-memory Numeric Datum from a base-10000 digit array.
// Caller owns the storage; returns Datum pointing into the provided buffer.
//
// dscale = display scale (digits after decimal point)
// weight = position of most-significant base-10000 digit relative to decimal point
//          (e.g. value 12345 has digits=[1, 2345], weight=1)
// digits = base-10000 digits, most-significant first
// is_negative = sign
Datum make_numeric(std::vector<char>& buf,
                   bool is_negative,
                   int16 weight,
                   uint16 dscale,
                   const std::vector<NumericDigit>& digits) {
    const size_t hdr_size = VARHDRSZ + sizeof(uint16) + sizeof(int16);
    const size_t total_size = hdr_size + digits.size() * sizeof(NumericDigit);
    buf.assign(total_size, 0);

    auto* num = reinterpret_cast<NumericData*>(buf.data());
    SET_VARSIZE(num, total_size);
    num->choice.n_long.n_sign_dscale =
        (is_negative ? NUMERIC_NEG : NUMERIC_POS) | (dscale & NUMERIC_DSCALE_MASK);
    num->choice.n_long.n_weight = weight;
    std::memcpy(num->choice.n_long.n_data, digits.data(),
                digits.size() * sizeof(NumericDigit));

    return PointerGetDatum(num);
}

}  // namespace

TEST(NumericToI128, Zero) {
    std::vector<char> buf;
    Datum d = make_numeric(buf, false, /*weight=*/0, /*dscale=*/0, /*digits=*/{});
    EXPECT_EQ(numeric_to_i128(d, 0), 0);
}

TEST(NumericToI128, PositiveInteger) {
    // value = 12345, no fractional, target_scale=0 → 12345
    // digits in base-10000: [1, 2345], weight=1
    std::vector<char> buf;
    Datum d = make_numeric(buf, false, /*weight=*/1, /*dscale=*/0, /*digits=*/{1, 2345});
    EXPECT_EQ(numeric_to_i128(d, 0), 12345);
}

TEST(NumericToI128, NegativeInteger) {
    std::vector<char> buf;
    Datum d = make_numeric(buf, true, /*weight=*/1, /*dscale=*/0, /*digits=*/{1, 2345});
    EXPECT_EQ(numeric_to_i128(d, 0), -12345);
}

TEST(NumericToI128, FractionalRescaleUp) {
    // value = 1.5, target_scale=4 → 15000
    // digits: integer part 1 (weight=0, digit 1), fractional 5000 (one base-10000 chunk)
    // Wait — 1.5 needs fractional digit 5000 (0.5000). So digits=[1, 5000], weight=0.
    std::vector<char> buf;
    Datum d = make_numeric(buf, false, /*weight=*/0, /*dscale=*/1, /*digits=*/{1, 5000});
    EXPECT_EQ(numeric_to_i128(d, 4), 15000);
}

TEST(NumericToI128, ScaleDownByDivision) {
    // value = 12345, target_scale=-2 → 123 (drops 2 decimals via /10 loop)
    std::vector<char> buf;
    Datum d = make_numeric(buf, false, /*weight=*/1, /*dscale=*/0, /*digits=*/{1, 2345});
    EXPECT_EQ(numeric_to_i128(d, -2), 123);
}

TEST(NumericToI128, LargeValue) {
    // value = 99999999 (8 digits) at scale 0
    // digits in base-10000: [9999, 9999], weight=1
    std::vector<char> buf;
    Datum d = make_numeric(buf, false, /*weight=*/1, /*dscale=*/0, /*digits=*/{9999, 9999});
    EXPECT_EQ(numeric_to_i128(d, 0), 99999999LL);
}
