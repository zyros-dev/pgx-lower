#include "core/error_handling.h"
#include "core/mlir_logger.h"

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

const char* ErrorInfo::getSeverityString() const {
    switch (severity) {
        case ErrorSeverity::INFO_LEVEL: return "INFO";
        case ErrorSeverity::WARNING_LEVEL: return "WARNING";
        case ErrorSeverity::ERROR_LEVEL: return "ERROR";
        case ErrorSeverity::FATAL_LEVEL: return "FATAL";
        default: return "UNKNOWN";
    }
}

const char* ErrorInfo::getCategoryString() const {
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

std::string ErrorInfo::getFormattedMessage() const {
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
    int pgLevel;
    
    switch (error.severity) {
        case ErrorSeverity::INFO_LEVEL:
            pgLevel = NOTICE;
            break;
        case ErrorSeverity::WARNING_LEVEL:
            pgLevel = WARNING;
            break;
        case ErrorSeverity::ERROR_LEVEL:
            pgLevel = ERROR;
            break;
        case ErrorSeverity::FATAL_LEVEL:
            pgLevel = FATAL;
            break;
        default:
            pgLevel = ERROR;
            break;
    }
    
    elog(pgLevel, "%s", error.getFormattedMessage().c_str());
}

bool PostgreSQLErrorHandler::shouldAbortOnError(const ErrorInfo& error) const {
    return error.severity == ErrorSeverity::FATAL_LEVEL;
}

#else

void PostgreSQLErrorHandler::handleError(const ErrorInfo& error) {
    // Fallback to console when not in PostgreSQL extension
    std::cerr << "PostgreSQL: " << error.getFormattedMessage() << std::endl;
}

bool PostgreSQLErrorHandler::shouldAbortOnError(const ErrorInfo& error) const {
    return error.severity == ErrorSeverity::FATAL_LEVEL;
}

#endif

// ===== ConsoleErrorHandler =====

void ConsoleErrorHandler::handleError(const ErrorInfo& error) {
    std::ostream& stream = (error.severity >= ErrorSeverity::ERROR_LEVEL) ? std::cerr : std::cout;
    stream << error.getFormattedMessage() << std::endl;
}

bool ConsoleErrorHandler::shouldAbortOnError(const ErrorInfo& error) const {
    return error.severity == ErrorSeverity::FATAL_LEVEL;
}

// ===== ErrorManager =====

std::unique_ptr<ErrorHandler> ErrorManager::handler_;

void ErrorManager::setHandler(std::unique_ptr<ErrorHandler> handler) {
    handler_ = std::move(handler);
}

ErrorHandler* ErrorManager::getHandler() {
    if (!handler_) {
        // Default to console handler if none set
        handler_ = std::make_unique<ConsoleErrorHandler>();
    }
    return handler_.get();
}

void ErrorManager::reportError(const ErrorInfo& error) {
    auto* handler = getHandler();
    if (handler) {
        handler->handleError(error);
        
        if (handler->shouldAbortOnError(error)) {
            throw std::runtime_error("Fatal error: " + error.message);
        }
    }
}

ErrorInfo ErrorManager::makeError(ErrorSeverity severity, ErrorCategory category,
                                 const std::string& message) {
    return ErrorInfo(severity, category, message);
}

ErrorInfo ErrorManager::makeError(ErrorSeverity severity, ErrorCategory category,
                                 const std::string& message, const std::string& context) {
    return ErrorInfo(severity, category, message, context);
}

ErrorInfo ErrorManager::queryAnalysisError(const std::string& message, 
                                          const std::string& queryText) {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::QUERY_ANALYSIS, message);
    if (!queryText.empty()) {
        error.context = "Query: " + queryText;
    }
    error.suggestion = "Check query syntax and ensure it uses supported SQL features";
    return error;
}

ErrorInfo ErrorManager::mlirGenerationError(const std::string& message,
                                           const std::string& context) {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::MLIR_GENERATION, message);
    error.context = context;
    error.suggestion = "This may indicate a bug in MLIR code generation logic";
    return error;
}

ErrorInfo ErrorManager::compilationError(const std::string& message,
                                        const std::string& context) {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::COMPILATION, message);
    error.context = context;
    error.suggestion = "Check LLVM/MLIR version compatibility and compilation flags";
    return error;
}

ErrorInfo ErrorManager::executionError(const std::string& message,
                                      const std::string& context) {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::EXECUTION, message);
    error.context = context;
    error.suggestion = "Check input data and runtime environment";
    return error;
}

ErrorInfo ErrorManager::postgresqlError(const std::string& message,
                                       const std::string& context) {
    auto error = ErrorInfo(ErrorSeverity::ERROR_LEVEL, ErrorCategory::POSTGRESQL, message);
    error.context = context;
    error.suggestion = "Check PostgreSQL version and configuration";
    return error;
}

} // namespace pgx_lower