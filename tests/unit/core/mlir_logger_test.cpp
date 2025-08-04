#include <gtest/gtest.h>
#include <execution/mlir_logger.h>
#include <sstream>

class MLIRLoggerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup for each test
    }
};

TEST_F(MLIRLoggerTest, ConsoleLoggerBasicLogging) {
    ConsoleLogger logger;

    // Test debug logging
    EXPECT_NO_THROW(logger.debug("Test debug message"));

    // Test notice logging
    EXPECT_NO_THROW(logger.notice("Test notice message"));

    // Test multiple messages
    EXPECT_NO_THROW(logger.debug("Debug 1"));
    EXPECT_NO_THROW(logger.debug("Debug 2"));
    EXPECT_NO_THROW(logger.notice("Notice 1"));
}

TEST_F(MLIRLoggerTest, LoggerWithDifferentMessageTypes) {
    ConsoleLogger logger;

    // Test with empty message
    EXPECT_NO_THROW(logger.debug(""));
    EXPECT_NO_THROW(logger.notice(""));

    // Test with long message
    std::string longMessage(1000, 'A');
    EXPECT_NO_THROW(logger.debug(longMessage));
    EXPECT_NO_THROW(logger.notice(longMessage));

    // Test with special characters
    EXPECT_NO_THROW(logger.debug("Message with special chars: !@#$%^&*()"));
    EXPECT_NO_THROW(logger.notice("Message with newlines\nand\ttabs"));
}

TEST_F(MLIRLoggerTest, LoggerPerformance) {
    ConsoleLogger logger;

    // Test logging many messages doesn't crash
    for (int i = 0; i < 100; ++i) {
        logger.debug("Debug message " + std::to_string(i));
        if (i % 10 == 0) {
            logger.notice("Notice message " + std::to_string(i));
        }
    }

    // Should complete without issues
    SUCCEED();
}