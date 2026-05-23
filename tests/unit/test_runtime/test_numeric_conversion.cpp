#include <gtest/gtest.h>
#include <cstring>
#include <vector>

extern "C" {
#include "postgres.h"
}

#include "pgx-lower/runtime/NumericConversion.h"

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

TEST(NumericToI128, Zero) {
    std::vector<char> buf;
    EXPECT_EQ(numeric_to_i128(make_numeric(buf, false, 0, 0, {}), 0), 0);
}

TEST(NumericToI128, PositiveInteger) {
    std::vector<char> buf;
    EXPECT_EQ(numeric_to_i128(make_numeric(buf, false, 1, 0, {1, 2345}), 0), 12345);
}

TEST(NumericToI128, NegativeInteger) {
    std::vector<char> buf;
    EXPECT_EQ(numeric_to_i128(make_numeric(buf, true, 1, 0, {1, 2345}), 0), -12345);
}

TEST(NumericToI128, FractionalRescaleUp) {
    std::vector<char> buf;
    EXPECT_EQ(numeric_to_i128(make_numeric(buf, false, 0, 1, {1, 5000}), 4), 15000);
}

TEST(NumericToI128, ScaleDownByDivision) {
    std::vector<char> buf;
    EXPECT_EQ(numeric_to_i128(make_numeric(buf, false, 1, 0, {1, 2345}), -2), 123);
}

TEST(NumericToI128, LargeValue) {
    std::vector<char> buf;
    EXPECT_EQ(numeric_to_i128(make_numeric(buf, false, 1, 0, {9999, 9999}), 0), 99999999LL);
}
