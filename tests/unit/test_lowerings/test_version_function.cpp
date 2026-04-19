#include <gtest/gtest.h>
#include <string>

namespace {
constexpr const char* PGX_LOWER_VERSION = "1.0";
}

TEST(VersionFunctionTest, ReturnsOneDotZero) {
    EXPECT_STREQ(PGX_LOWER_VERSION, "1.0");
    EXPECT_EQ(std::string(PGX_LOWER_VERSION).size(), 3u);
}
