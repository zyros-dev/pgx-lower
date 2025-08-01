#include <gtest/gtest.h>
#include "../test_helpers.h"

class TupleAccessTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup for each test
    }

    void TearDown() override {
        // Cleanup mock context
        g_mock_scan_context = nullptr;
    }
};

TEST_F(TupleAccessTest, MockGetNextTuple) {
    const std::vector<int64_t> mockData = {1, 2, 3};
    MockTupleScanContext mockContext = {mockData, 0, true};
    // ReSharper disable once CppDFALocalValueEscapesFunction
    g_mock_scan_context = &mockContext;

    // Test getting tuples sequentially
    EXPECT_EQ(mock_get_next_tuple(), 1);
    EXPECT_EQ(mock_get_next_tuple(), 2);
    EXPECT_EQ(mock_get_next_tuple(), 3);

    // Test end of data
    EXPECT_EQ(mock_get_next_tuple(), -2);
}

TEST_F(TupleAccessTest, MockTableOperations) {
    const std::vector<int64_t> mockData = {100, 200};
    MockTupleScanContext mockContext = {mockData, 0, true};
    // ReSharper disable once CppDFALocalValueEscapesFunction
    g_mock_scan_context = &mockContext;

    // Test opening table
    int64_t handle = open_postgres_table(12345);
    EXPECT_EQ(handle, reinterpret_cast<int64_t>(&mockContext));

    // Test reading tuples
    EXPECT_EQ(read_next_tuple_from_table(handle), 100);
    EXPECT_EQ(read_next_tuple_from_table(handle), 200);
    EXPECT_EQ(read_next_tuple_from_table(handle), -2);

    // Test closing table (should not crash)
    close_postgres_table(handle);
}

TEST_F(TupleAccessTest, MockFieldAccess) {
    // Test integer field access
    bool is_null = true;
    int32_t int_value = get_int_field(12345, 2, &is_null);
    EXPECT_FALSE(is_null);
    EXPECT_EQ(int_value, 2 * 42); // field_index * 42

    // Test text field access
    is_null = true;
    int64_t text_ptr = get_text_field(12345, 1, &is_null);
    EXPECT_FALSE(is_null);
    EXPECT_NE(text_ptr, 0);

    const char* text = reinterpret_cast<const char*>(text_ptr);
    EXPECT_STREQ(text, "mock_text_field");
}

TEST_F(TupleAccessTest, MockAddTupleToResult) {
    // Test adding tuple to result
    EXPECT_TRUE(add_tuple_to_result(12345));
    EXPECT_TRUE(add_tuple_to_result(67890));
}

TEST_F(TupleAccessTest, EdgeCases) {
    // Test with null context
    g_mock_scan_context = nullptr;

    EXPECT_EQ(mock_get_next_tuple(), -1);
    EXPECT_EQ(open_postgres_table(12345), 0);
    EXPECT_EQ(read_next_tuple_from_table(0), -1);

    // Test empty data
    constexpr std::vector<int64_t> emptyData = {};
    MockTupleScanContext emptyContext = {emptyData, 0, true};
    // ReSharper disable once CppDFALocalValueEscapesFunction
    g_mock_scan_context = &emptyContext;

    EXPECT_EQ(mock_get_next_tuple(), -2);
}