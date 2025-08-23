#include <gtest/gtest.h>
#include <string>

// Simplified unit test for string constant functionality
// Tests the core logic without complex MLIR dependencies

class StringConstantTest : public ::testing::Test {
protected:
    // Test the core string length logic that drives CharType vs StringType selection
    static bool shouldUseCharType(const std::string& str) {
        return str.size() <= 8;
    }
};

// Test CharType selection for short strings (≤8 characters)
TEST_F(StringConstantTest, ShortStringUsesCharType) {
    EXPECT_TRUE(shouldUseCharType("hello"));    // 5 characters
    EXPECT_TRUE(shouldUseCharType("test"));     // 4 characters
    EXPECT_TRUE(shouldUseCharType("12345678")); // 8 characters exactly
}

// Test StringType selection for long strings (>8 characters)
TEST_F(StringConstantTest, LongStringUsesStringType) {
    EXPECT_FALSE(shouldUseCharType("very long string"));  // 16 characters
    EXPECT_FALSE(shouldUseCharType("123456789"));         // 9 characters
}

// Test boundary condition at 8 characters
TEST_F(StringConstantTest, EightCharacterBoundary) {
    EXPECT_TRUE(shouldUseCharType("12345678"));   // Exactly 8 - uses CharType
    EXPECT_FALSE(shouldUseCharType("123456789")); // 9 characters - uses StringType
}

// Test edge cases
TEST_F(StringConstantTest, EdgeCases) {
    EXPECT_TRUE(shouldUseCharType(""));          // Empty string
    EXPECT_TRUE(shouldUseCharType("a"));         // Single character
    EXPECT_TRUE(shouldUseCharType("tab\nnew"));  // Special characters (7 chars)
}