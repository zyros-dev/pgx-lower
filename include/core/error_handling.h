#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pgx_lower {

enum class ErrorSeverity : std::uint8_t {
    INFO_LEVEL,
    WARNING_LEVEL,
    ERROR_LEVEL,
    FATAL_LEVEL
};

enum class ErrorCategory : std::uint8_t {
    INITIALIZATION,
    QUERY_ANALYSIS,
    MLIR_GENERATION,
    COMPILATION,
    EXECUTION,
    POSTGRESQL,
    MEMORY,
    IO
};

struct ErrorInfo {
    ErrorSeverity severity;
    ErrorCategory category;
    std::string message;
    std::string context;
    int errorCode = 0;
    std::string suggestion;

    ErrorInfo(const ErrorSeverity sev, const ErrorCategory cat, std::string  msg)
    : severity(sev)
    , category(cat)
    , message(std::move(msg)) {}

    ErrorInfo(const ErrorSeverity sev, const ErrorCategory cat, std::string msg, std::string  ctx)
    : severity(sev)
    , category(cat)
    , message(std::move(msg))
    , context(std::move(ctx)) {}

    // Get human-readable severity string
    [[nodiscard]] auto getSeverityString() const -> const char*;

    // Get human-readable category string
    [[nodiscard]] auto getCategoryString() const -> const char*;

    // Get formatted error message
    [[nodiscard]] auto getFormattedMessage() const -> std::string;
};

/**
 * Result type for operations that can fail
 */
template<typename T>
class Result {
   private:
    bool success_;
    T value_;
    ErrorInfo error_;

   public:
    // Success constructor
    explicit Result(T&& value)
    : success_(true)
    , value_(std::move(value))
    , error_(ErrorSeverity::INFO_LEVEL, ErrorCategory::EXECUTION, "") {}

    explicit Result(const T& value)
    : success_(true)
    , value_(value)
    , error_(ErrorSeverity::INFO_LEVEL, ErrorCategory::EXECUTION, "") {}

    // Error constructor
    explicit Result(ErrorInfo  error)
    : success_(false)
    , value_()
    , error_(std::move(error)) {}

    // Check if operation succeeded
    [[nodiscard]] auto isSuccess() const -> bool { return success_; }
    [[nodiscard]] auto isError() const -> bool { return !success_; }

    // Get value (only valid if isSuccess())
    auto getValue() const -> const T& {
        if (!success_) {
            throw std::runtime_error("Attempted to get value from failed Result");
        }
        return value_;
    }

    auto getValue() -> T& {
        if (!success_) {
            throw std::runtime_error("Attempted to get value from failed Result");
        }
        return value_;
    }

    // Get error (only valid if isError())
    [[nodiscard]] auto getError() const -> const ErrorInfo& {
        if (success_) {
            throw std::runtime_error("Attempted to get error from successful Result");
        }
        return error_;
    }

    // Convenience methods
    auto valueOr(const T& defaultValue) const -> T { return success_ ? value_ : defaultValue; }
};

/**
 * Result type for operations that don't return a value
 */
using VoidResult = Result<bool>;

/**
 * Error handler interface for different environments
 */
class ErrorHandler {
   public:
    virtual ~ErrorHandler() = default;

    // Handle an error
    virtual void handleError(const ErrorInfo& error) = 0;

    // Check if errors should be treated as fatal
    [[nodiscard]] virtual auto shouldAbortOnError(const ErrorInfo& error) const -> bool = 0;
};

/**
 * PostgreSQL-specific error handler
 */
class PostgreSQLErrorHandler : public ErrorHandler {
   public:
    void handleError(const ErrorInfo& error) override;
    [[nodiscard]] auto shouldAbortOnError(const ErrorInfo& error) const -> bool override;
};

/**
 * Console-based error handler for unit tests
 */
class ConsoleErrorHandler : public ErrorHandler {
   public:
    void handleError(const ErrorInfo& error) override;
    [[nodiscard]] auto shouldAbortOnError(const ErrorInfo& error) const -> bool override;
};

/**
 * Global error handling utilities
 */
class ErrorManager {
   private:
    static std::unique_ptr<ErrorHandler> handler_;

   public:
    // Set the global error handler
    static void setHandler(std::unique_ptr<ErrorHandler> handler);

    // Get the current error handler
    static auto getHandler() -> ErrorHandler*;

    // Report an error through the current handler
    static void reportError(const ErrorInfo& error);

    // Convenience methods for creating errors
    static auto makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message) -> ErrorInfo;

    static auto
    makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message, const std::string& context) -> ErrorInfo;

    // Create specific error types
    static auto queryAnalysisError(const std::string& message, const std::string& queryText = "") -> ErrorInfo;

    static auto mlirGenerationError(const std::string& message, const std::string& context = "") -> ErrorInfo;

    static auto compilationError(const std::string& message, const std::string& context = "") -> ErrorInfo;

    static auto executionError(const std::string& message, const std::string& context = "") -> ErrorInfo;

    static auto postgresqlError(const std::string& message, const std::string& context = "") -> ErrorInfo;
};

// Convenience macros for error creation
#define MAKE_ERROR(severity, category, message)                                                                        \
    pgx_lower::ErrorManager::makeError(pgx_lower::ErrorSeverity::severity, pgx_lower::ErrorCategory::category, message)

#define MAKE_ERROR_CTX(severity, category, message, context)                                                           \
    pgx_lower::ErrorManager::makeError(pgx_lower::ErrorSeverity::severity,                                             \
                                       pgx_lower::ErrorCategory::category,                                             \
                                       message,                                                                        \
                                       context)

#define REPORT_ERROR(severity, category, message)                                                                      \
    pgx_lower::ErrorManager::reportError(MAKE_ERROR(severity, category, message))

#define REPORT_ERROR_CTX(severity, category, message, context)                                                         \
    pgx_lower::ErrorManager::reportError(MAKE_ERROR_CTX(severity, category, message, context))

} // namespace pgx_lower