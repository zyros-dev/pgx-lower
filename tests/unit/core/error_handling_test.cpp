#include <gtest/gtest.h>
#include <execution/error_handling.h>

class ErrorHandlingTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup for each test
    }
};

TEST_F(ErrorHandlingTest, ErrorCreationAndFormatting) {
    using namespace pgx_lower;

    // Test error creation and formatting
    const auto error = ErrorManager::queryAnalysisError("Test error message", "SELECT * FROM test");
    EXPECT_EQ(error.severity, ErrorSeverity::ERROR_LEVEL);
    EXPECT_EQ(error.category, ErrorCategory::QUERY_ANALYSIS);
    EXPECT_EQ(error.message, "Test error message");
    EXPECT_TRUE(error.context.find("SELECT * FROM test") != std::string::npos);

    // Test formatted message
    std::string formatted = error.getFormattedMessage();
    EXPECT_TRUE(formatted.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(formatted.find("[QUERY_ANALYSIS]") != std::string::npos);
    EXPECT_TRUE(formatted.find("Test error message") != std::string::npos);
}

TEST_F(ErrorHandlingTest, ResultType) {
    using namespace pgx_lower;

    // Test Result type for success
    auto successResult = Result(42);
    EXPECT_TRUE(successResult.isSuccess());
    EXPECT_FALSE(successResult.isError());
    EXPECT_EQ(successResult.getValue(), 42);

    // Test Result type for error
    auto errorInfo = ErrorManager::makeError(ErrorSeverity::ERROR_LEVEL, ErrorCategory::EXECUTION, "Test failure");
    auto errorResult = Result<int>(errorInfo);
    EXPECT_FALSE(errorResult.isSuccess());
    EXPECT_TRUE(errorResult.isError());
    EXPECT_EQ(errorResult.valueOr(99), 99);
}

TEST_F(ErrorHandlingTest, ErrorHandler) {
    using namespace pgx_lower;

    // Test console error handler
    ErrorManager::setHandler(std::make_unique<ConsoleErrorHandler>());
    const auto handler = ErrorManager::getHandler();
    EXPECT_NE(handler, nullptr);

    const auto error = ErrorManager::queryAnalysisError("Test error", "SELECT * FROM test");
    EXPECT_FALSE(handler->shouldAbortOnError(error));

    // Test fatal error should abort
    const auto fatalError = ErrorManager::makeError(ErrorSeverity::FATAL_LEVEL, ErrorCategory::EXECUTION, "Fatal error");
    EXPECT_TRUE(handler->shouldAbortOnError(fatalError));
}