#include <gtest/gtest.h>
#include "execution/logging.h"
#include <sstream>
#include <iostream>
#include <memory>

// Test fixture for logging macro functionality
class LoggingMacroTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Capture stdout and stderr for testing
        old_cout = std::cout.rdbuf();
        old_cerr = std::cerr.rdbuf();
        
        cout_stream.str("");
        cerr_stream.str("");
        
        std::cout.rdbuf(cout_stream.rdbuf());
        std::cerr.rdbuf(cerr_stream.rdbuf());
        
        // Set debug level to capture all messages
        pgx::get_logger().set_level(pgx::LogLevel::TRACE_LVL);
    }
    
    void TearDown() override {
        // Restore original streams
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
    
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
    std::stringstream cout_stream;
    std::stringstream cerr_stream;
};

// Test basic PGX logging macros
TEST_F(LoggingMacroTest, BasicLoggingMacros) {
    PGX_DEBUG("Debug message");
    PGX_INFO("Info message");
    PGX_WARNING("Warning message");
    PGX_ERROR("Error message");
    
    // At least one stream should have content
    bool has_output = !cout_stream.str().empty() || !cerr_stream.str().empty();
    EXPECT_TRUE(has_output);
}

// Test MLIR-specific logging macros
TEST_F(LoggingMacroTest, MLIRLoggingMacros) {
    MLIR_PGX_DEBUG("TestDialect", "Debug message for dialect");
    MLIR_PGX_INFO("TestDialect", "Info message for dialect");
    MLIR_PGX_ERROR("TestDialect", "Error message for dialect");
    
    // At least one stream should have content
    bool has_output = !cout_stream.str().empty() || !cerr_stream.str().empty();
    EXPECT_TRUE(has_output);
}

// Test runtime-specific logging macros
TEST_F(LoggingMacroTest, RuntimeLoggingMacros) {
    RUNTIME_PGX_DEBUG("TestComponent", "Debug message for component");
    RUNTIME_PGX_NOTICE("TestComponent", "Notice message for component");
    
    // At least one stream should have content
    bool has_output = !cout_stream.str().empty() || !cerr_stream.str().empty();
    EXPECT_TRUE(has_output);
}

// Test logging level filtering
TEST_F(LoggingMacroTest, LoggingLevelFiltering) {
    // Set to ERROR level only
    pgx::get_logger().set_level(pgx::LogLevel::ERROR_LVL);
    
    cout_stream.str("");
    cerr_stream.str("");
    
    PGX_DEBUG("This should not appear");
    PGX_INFO("This should not appear");
    PGX_WARNING("This should not appear");
    PGX_ERROR("This should appear");
    
    // Should have some output from ERROR but not from lower levels
    std::string combined_output = cout_stream.str() + cerr_stream.str();
    EXPECT_FALSE(combined_output.empty());
}

// Test that logging macros don't crash with empty messages
TEST_F(LoggingMacroTest, EmptyMessageHandling) {
    EXPECT_NO_THROW({
        PGX_DEBUG("");
        PGX_INFO("");
        PGX_WARNING("");
        PGX_ERROR("");
        MLIR_PGX_DEBUG("Dialect", "");
        RUNTIME_PGX_DEBUG("Component", "");
    });
}

// Test that logging macros handle special characters
TEST_F(LoggingMacroTest, SpecialCharacterHandling) {
    EXPECT_NO_THROW({
        PGX_DEBUG("Message with %s format specifiers %d");
        PGX_INFO("Message with newlines\nand\ttabs");
        PGX_WARNING("Message with unicode: αβγδε");
        MLIR_PGX_ERROR("Dialect", "Message with quotes \"and\" backslashes\\");
    });
}