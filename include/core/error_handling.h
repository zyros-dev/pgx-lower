#pragma once

#include <string>
#include <memory>
#include <stdexcept>

namespace pgx_lower {

/**
 * Error severity levels for categorizing issues
 */
enum class ErrorSeverity {
    INFO_LEVEL, // Informational - operation succeeded with notes
    WARNING_LEVEL, // Warning - operation succeeded but with concerns
    ERROR_LEVEL, // Error - operation failed but system can continue
    FATAL_LEVEL // Fatal - operation failed and system should stop
};

/**
 * Error categories to help with debugging and monitoring
 */
enum class ErrorCategory {
    INITIALIZATION, // Startup/initialization errors
    QUERY_ANALYSIS, // Query parsing and analysis errors
    MLIR_GENERATION, // MLIR code generation errors
    COMPILATION, // LLVM/JIT compilation errors
    EXECUTION, // Runtime execution errors
    POSTGRESQL, // PostgreSQL integration errors
    MEMORY, // Memory allocation/management errors
    IO // Input/output errors
};

/**
 * Structured error information
 */
struct ErrorInfo {
    ErrorSeverity severity;
    ErrorCategory category;
    std::string message;
    std::string context; // Additional context (function name, query text, etc.)
    int errorCode = 0; // Numeric error code for programmatic handling
    std::string suggestion; // Suggested fix or workaround

    ErrorInfo(ErrorSeverity sev, ErrorCategory cat, const std::string& msg)
    : severity(sev)
    , category(cat)
    , message(msg) {}

    ErrorInfo(ErrorSeverity sev, ErrorCategory cat, const std::string& msg, const std::string& ctx)
    : severity(sev)
    , category(cat)
    , message(msg)
    , context(ctx) {}

    // Get human-readable severity string
    const char* getSeverityString() const;

    // Get human-readable category string
    const char* getCategoryString() const;

    // Get formatted error message
    std::string getFormattedMessage() const;
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
    explicit Result(const ErrorInfo& error)
    : success_(false)
    , value_()
    , error_(error) {}

    // Check if operation succeeded
    bool isSuccess() const { return success_; }
    bool isError() const { return !success_; }

    // Get value (only valid if isSuccess())
    const T& getValue() const {
        if (!success_) {
            throw std::runtime_error("Attempted to get value from failed Result");
        }
        return value_;
    }

    T& getValue() {
        if (!success_) {
            throw std::runtime_error("Attempted to get value from failed Result");
        }
        return value_;
    }

    // Get error (only valid if isError())
    const ErrorInfo& getError() const {
        if (success_) {
            throw std::runtime_error("Attempted to get error from successful Result");
        }
        return error_;
    }

    // Convenience methods
    T valueOr(const T& defaultValue) const { return success_ ? value_ : defaultValue; }
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
    virtual bool shouldAbortOnError(const ErrorInfo& error) const = 0;
};

/**
 * PostgreSQL-specific error handler
 */
class PostgreSQLErrorHandler : public ErrorHandler {
   public:
    void handleError(const ErrorInfo& error) override;
    bool shouldAbortOnError(const ErrorInfo& error) const override;
};

/**
 * Console-based error handler for unit tests
 */
class ConsoleErrorHandler : public ErrorHandler {
   public:
    void handleError(const ErrorInfo& error) override;
    bool shouldAbortOnError(const ErrorInfo& error) const override;
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
    static ErrorHandler* getHandler();

    // Report an error through the current handler
    static void reportError(const ErrorInfo& error);

    // Convenience methods for creating errors
    static ErrorInfo makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message);

    static ErrorInfo
    makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message, const std::string& context);

    // Create specific error types
    static ErrorInfo queryAnalysisError(const std::string& message, const std::string& queryText = "");

    static ErrorInfo mlirGenerationError(const std::string& message, const std::string& context = "");

    static ErrorInfo compilationError(const std::string& message, const std::string& context = "");

    static ErrorInfo executionError(const std::string& message, const std::string& context = "");

    static ErrorInfo postgresqlError(const std::string& message, const std::string& context = "");
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