#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>
#include <utility>

// I like the idea of this class, but I just started using std::runtime_error instead...
// In the future I should probably refactor those to use this class, or something like this...
// You can tell it looks very similiar to our logging system.
namespace pgx_lower {

enum class ErrorSeverity : std::uint8_t { INFO_LEVEL, WARNING_LEVEL, ERROR_LEVEL, FATAL_LEVEL };

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

    ErrorInfo(const ErrorSeverity sev, const ErrorCategory cat, std::string msg)
    : severity(sev)
    , category(cat)
    , message(std::move(msg)) {}

    ErrorInfo(const ErrorSeverity sev, const ErrorCategory cat, std::string msg, std::string ctx)
    : severity(sev)
    , category(cat)
    , message(std::move(msg))
    , context(std::move(ctx)) {}

    [[nodiscard]] auto getSeverityString() const -> const char*;

    [[nodiscard]] auto getCategoryString() const -> const char*;

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
    explicit Result(T&& value)
    : success_(true)
    , value_(std::move(value))
    , error_(ErrorSeverity::INFO_LEVEL, ErrorCategory::EXECUTION, "") {}

    explicit Result(const T& value)
    : success_(true)
    , value_(value)
    , error_(ErrorSeverity::INFO_LEVEL, ErrorCategory::EXECUTION, "") {}

    explicit Result(ErrorInfo error)
    : success_(false)
    , value_()
    , error_(std::move(error)) {}

    [[nodiscard]] auto isSuccess() const -> bool { return success_; }
    [[nodiscard]] auto isError() const -> bool { return !success_; }

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

    [[nodiscard]] auto getError() const -> const ErrorInfo& {
        if (success_) {
            throw std::runtime_error("Attempted to get error from successful Result");
        }
        return error_;
    }

    auto valueOr(const T& defaultValue) const -> T { return success_ ? value_ : defaultValue; }
};

using VoidResult = Result<bool>;

class ErrorHandler {
   public:
    virtual ~ErrorHandler() = default;

    virtual void handleError(const ErrorInfo& error) = 0;

    [[nodiscard]] virtual auto shouldAbortOnError(const ErrorInfo& error) const -> bool = 0;
};

class PostgreSQLErrorHandler : public ErrorHandler {
   public:
    void handleError(const ErrorInfo& error) override;
    [[nodiscard]] auto shouldAbortOnError(const ErrorInfo& error) const -> bool override;
};

class ConsoleErrorHandler : public ErrorHandler {
   public:
    void handleError(const ErrorInfo& error) override;
    [[nodiscard]] auto shouldAbortOnError(const ErrorInfo& error) const -> bool override;
};

class ErrorManager {
   private:
    static std::unique_ptr<ErrorHandler> handler_;

   public:
    static void setHandler(std::unique_ptr<ErrorHandler> handler);

    static auto getHandler() -> ErrorHandler*;

    static void reportError(const ErrorInfo& error);

    static auto makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message) -> ErrorInfo;

    static auto makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message,
                          const std::string& context) -> ErrorInfo;

    static auto queryAnalysisError(const std::string& message, const std::string& queryText = "") -> ErrorInfo;
    static auto mlirGenerationError(const std::string& message, const std::string& context = "") -> ErrorInfo;
    static auto compilationError(const std::string& message, const std::string& context = "") -> ErrorInfo;
    static auto executionError(const std::string& message, const std::string& context = "") -> ErrorInfo;
    static auto postgresqlError(const std::string& message, const std::string& context = "") -> ErrorInfo;
};

#define MAKE_ERROR(severity, category, message)                                                                        \
    pgx_lower::ErrorManager::makeError(pgx_lower::ErrorSeverity::severity, pgx_lower::ErrorCategory::category, message)

#define MAKE_ERROR_CTX(severity, category, message, context)                                                           \
    pgx_lower::ErrorManager::makeError(pgx_lower::ErrorSeverity::severity, pgx_lower::ErrorCategory::category,         \
                                       message, context)

#define REPORT_ERROR(severity, category, message)                                                                      \
    pgx_lower::ErrorManager::reportError(MAKE_ERROR(severity, category, message))

#define REPORT_ERROR_CTX(severity, category, message, context)                                                         \
    pgx_lower::ErrorManager::reportError(MAKE_ERROR_CTX(severity, category, message, context))

} // namespace pgx_lower