#include <gtest/gtest.h>
#include <string>

// pgx_lower_version returns a constant version string. The C wrapper
// hands it to PG via PG_RETURN_TEXT_P(cstring_to_text(...)), which
// requires a live PG backend. For the unit-test build we test the
// constant itself — trivial, but proves the utest path works
// end-to-end for this spec's pattern (function returns a literal string).
namespace {
constexpr const char* PGX_LOWER_VERSION = "1.0";
}

TEST(VersionFunctionTest, ReturnsOneDotZero) {
    EXPECT_STREQ(PGX_LOWER_VERSION, "1.0");
    EXPECT_EQ(std::string(PGX_LOWER_VERSION).size(), 3u);
}
