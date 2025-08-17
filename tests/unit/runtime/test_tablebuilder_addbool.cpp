#include <gtest/gtest.h>
#include <cstdlib>

// Declare the runtime function we're testing
extern "C" {
    void rt_tablebuilder_addbool(void* builder, bool is_null, bool value);
}

class TableBuilderBoolTest : public ::testing::Test {
protected:
    void* dummy_builder;
    
    void SetUp() override {
        // Create a dummy builder for testing
        dummy_builder = malloc(sizeof(int));
    }
    
    void TearDown() override {
        if (dummy_builder) {
            free(dummy_builder);
            dummy_builder = nullptr;
        }
    }
};

TEST_F(TableBuilderBoolTest, AddBoolWithNullFlag) {
    // Test that the function handles null values
    EXPECT_NO_THROW(rt_tablebuilder_addbool(dummy_builder, true, false));
}

TEST_F(TableBuilderBoolTest, AddBoolTrue) {
    // Test adding a true boolean value
    EXPECT_NO_THROW(rt_tablebuilder_addbool(dummy_builder, false, true));
}

TEST_F(TableBuilderBoolTest, AddBoolFalse) {
    // Test adding a false boolean value
    EXPECT_NO_THROW(rt_tablebuilder_addbool(dummy_builder, false, false));
}

TEST_F(TableBuilderBoolTest, AddBoolWithNullBuilder) {
    // Test that the function handles null builder gracefully
    EXPECT_NO_THROW(rt_tablebuilder_addbool(nullptr, false, true));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}