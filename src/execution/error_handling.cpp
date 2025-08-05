#include "execution/error_handling.h"
#include "execution/mlir_logger.h"
#include "execution/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}
#endif

#include <iostream>
#include <sstream>

namespace pgx_lower {

// ===== ErrorInfo =====

auto ErrorInfo::getSeverityString() const -> const char* {
    switch (severity) {
    case ErrorSeverity::INFO_LEVEL: return "INFO";
    case ErrorSeverity::WARNING_LEVEL: return "WARNING";
    case ErrorSeverity::ERROR_LEVEL: return "ERROR";
    case ErrorSeverity::FATAL_LEVEL: return "FATAL";
    default: return "UNKNOWN";
    }
}

auto ErrorInfo::getCategoryString() const -> const char* {
    switch (category) {
    case ErrorCategory::INITIALIZATION: return "INITIALIZATION";
    case ErrorCategory::QUERY_ANALYSIS: return "QUERY_ANALYSIS";
    case ErrorCategory::MLIR_GENERATION: return "MLIR_GENERATION";
    case ErrorCategory::COMPILATION: return "COMPILATION";
    case ErrorCategory::EXECUTION: return "EXECUTION";
    case ErrorCategory::POSTGRESQL: return "POSTGRESQL";
    case ErrorCategory::MEMORY: return "MEMORY";
    case ErrorCategory::IO: return "IO";
    default: return "UNKNOWN";
    }
}

auto ErrorInfo::getFormattedMessage() const -> std::string {
    std::ostringstream oss;
    oss << "[" << getSeverityString() << "][" << getCategoryString() << "] " << message;

    if (!context.empty()) {
        oss << " (Context: " << context << ")";
    }

    if (errorCode != 0) {
        oss << " (Code: " << errorCode << ")";
    }

    if (!suggestion.empty()) {
        oss << " - Suggestion: " << suggestion;
    }

    return oss.str();
}

// ===== PostgreSQLErrorHandler =====

#ifdef POSTGRESQL_EXTENSION

void PostgreSQLErrorHandler::handleError(const ErrorInfo& error) {
    switch (error.severity) {
    case ErrorSeverity::INFO_LEVEL: PGX_INFO(error.getFormattedMessage()); break;
    case ErrorSeverity::WARNING_LEVEL: PGX_WARNING(error.getFormattedMessage()); break;
    case ErrorSeverity::ERROR_LEVEL: PGX_ERROR(error.getFormattedMessage()); break;
    case ErrorSeverity::FATAL_LEVEL: PGX_ERROR(error.getFormattedMessage()); break;
    default: PGX_ERROR(error.getFormattedMessage()); break;
    }
}

bool PostgreSQLErrorHandler::shouldAbortOnError(const ErrorInfo& error) const {
    return error.severity == ErrorSeverity::FATAL_LEVEL;
}

#else

void PostgreSQLErrorHandler::handleError(const ErrorInfo& error) {
    // Fallback to PGX logging when not in PostgreSQL extension
    switch (error.severity) {
    case ErrorSeverity::INFO_LEVEL: PGX_INFO(error.getFormattedMessage()); break;
    case ErrorSeverity::WARNING_LEVEL: PGX_WARNING(error.getFormattedMessage()); break;
    case ErrorSeverity::ERROR_LEVEL: PGX_ERROR(error.getFormattedMessage()); break;
    case ErrorSeverity::FATAL_LEVEL: PGX_ERROR(error.getFormattedMessage()); break;
    default: PGX_ERROR(error.getFormattedMessage()); break;
    }
}

auto PostgreSQLErrorHandler::shouldAbortOnError(const ErrorInfo& error) const -> bool {
    return error.severity == ErrorSeverity::FATAL_LEVEL;
}

#endif

// ===== ConsoleErrorHandler =====

void ConsoleErrorHandler::handleError(const ErrorInfo& error) {
    switch (error.severity) {
    case ErrorSeverity::INFO_LEVEL: PGX_INFO(error.getFormattedMessage()); break;
    case ErrorSeverity::WARNING_LEVEL: PGX_WARNING(error.getFormattedMessage()); break;
    case ErrorSeverity::ERROR_LEVEL: PGX_ERROR(error.getFormattedMessage()); break;
    case ErrorSeverity::FATAL_LEVEL: PGX_ERROR(error.getFormattedMessage()); break;
    default: PGX_ERROR(error.getFormattedMessage()); break;
    }
}

auto ConsoleErrorHandler::shouldAbortOnError(const ErrorInfo& error) const -> bool {
    return error.severity == ErrorSeverity::FATAL_LEVEL;
}

// ===== ErrorManager =====

std::unique_ptr<ErrorHandler> ErrorManager::handler_;

void ErrorManager::setHandler(std::unique_ptr<ErrorHandler> handler) {
    handler_ = std::move(handler);
}

auto ErrorManager::getHandler() -> ErrorHandler* {
    if (!handler_) {
        // Default to console handler if none set
        handler_ = std::make_unique<ConsoleErrorHandler>();
    }
    return handler_.get();
}

void ErrorManager::reportError(const ErrorInfo& error) {
    auto* handler = getHandler();
    if (handler != nullptr) {
        handler->handleError(error);

        if (handler->shouldAbortOnError(error)) {
            throw std::runtime_error("Fatal error: " + error.message);
        }
    }
}

auto ErrorManager::makeError(ErrorSeverity severity, ErrorCategory category, const std::string& message) -> ErrorInfo {
    return {severity, category, message};
}

auto ErrorManager::makeError(ErrorSeverity severity,
                                  ErrorCategory category,
                                  const std::string& message,
                                  const std::string& context) -> ErrorInfo {
    return {severity, category, message, context};
}

auto ErrorManager::queryAnalysisError(const std::string& message, const std::string& queryText) -> ErrorInfo {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::QUERY_ANALYSIS, message);
    if (!queryText.empty()) {
        error.context = "Query: " + queryText;
    }
    error.suggestion = "Check query syntax and ensure it uses supported SQL features";
    return error;
}

auto ErrorManager::mlirGenerationError(const std::string& message, const std::string& context) -> ErrorInfo {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::MLIR_GENERATION, message);
    error.context = context;
    error.suggestion = "This may indicate a bug in MLIR code generation logic";
    return error;
}

auto ErrorManager::compilationError(const std::string& message, const std::string& context) -> ErrorInfo {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::COMPILATION, message);
    error.context = context;
    error.suggestion = "Check LLVM/MLIR version compatibility and compilation flags";
    return error;
}

auto ErrorManager::executionError(const std::string& message, const std::string& context) -> ErrorInfo {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::EXECUTION, message);
    error.context = context;
    error.suggestion = "Check input data and runtime environment";
    return error;
}

auto ErrorManager::postgresqlError(const std::string& message, const std::string& context) -> ErrorInfo {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::POSTGRESQL, message);
    error.context = context;
    error.suggestion = "Check PostgreSQL version and configuration";
    return error;
}

} // namespace pgx_lower