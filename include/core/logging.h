#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

// Add PostgreSQL integration when building as extension
#ifdef POSTGRESQL_EXTENSION
// Push/pop PostgreSQL macros to avoid conflicts
#pragma push_macro("_")
#pragma push_macro("gettext")
#pragma push_macro("dgettext") 
#pragma push_macro("ngettext")
#pragma push_macro("dngettext")
#pragma push_macro("dcgettext")

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

// Restore original macros
#pragma pop_macro("dcgettext")
#pragma pop_macro("dngettext")
#pragma pop_macro("ngettext")
#pragma pop_macro("dgettext")
#pragma pop_macro("gettext")
#pragma pop_macro("_")
#endif

namespace pgx {

enum class LogLevel {
    TRACE_LVL = 0,
    DEBUG_LVL = 1,
    INFO_LVL = 2,
    WARNING_LVL = 3,
    ERROR_LVL = 4
};

class Logger {
private:
    LogLevel current_level;
    std::ofstream debug_file;
    bool debug_to_file;
    
public:
    Logger();
    ~Logger();
    
    void set_level(LogLevel level);
    void set_debug_file(const std::string& filename);
    bool should_log(LogLevel level) const;
    void log(LogLevel level, const char* file, int line, const std::string& message);
};

extern Logger& get_logger();

// Convenient macros for logging
#define PGX_TRACE(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::TRACE_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::TRACE_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_DEBUG(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::DEBUG_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::DEBUG_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_INFO(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::INFO_LVL)) { \
            pgx::get_logger().log(pgx::LogLevel::INFO_LVL, __FILE__, __LINE__, msg); \
        } \
    } while (0)

#define PGX_WARNING(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::WARNING_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::WARNING_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_ERROR(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::ERROR_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::ERROR_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#ifdef POSTGRESQL_EXTENSION
#undef PGX_INFO
#define PGX_INFO(msg) \
    do { \
        elog(NOTICE, "[INFO] %s (%s:%d)", (msg), __FILE__, __LINE__); \
    } while (0)

#undef PGX_DEBUG
#define PGX_DEBUG(msg) \
    do { \
        elog(LOG, "[DEBUG] %s (%s:%d)", (msg), __FILE__, __LINE__); \
    } while (0)

#undef PGX_WARNING
#define PGX_WARNING(msg) \
    do { \
        elog(WARNING, "[WARNING] %s (%s:%d)", (msg), __FILE__, __LINE__); \
    } while (0)

#undef PGX_ERROR
#define PGX_ERROR(msg) \
    do { \
        elog(ERROR, "[ERROR] %s (%s:%d)", (msg), __FILE__, __LINE__); \
    } while (0)
#endif

#define PGX_NOTICE(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::INFO_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::INFO_LVL, __FILE__, __LINE__, msg); \
    } while (0)

// MLIR-specific logging macros
#define MLIR_PGX_DEBUG(dialect, msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::DEBUG_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            pgx::get_logger().log(pgx::LogLevel::DEBUG_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define MLIR_PGX_INFO(dialect, msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::INFO_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            pgx::get_logger().log(pgx::LogLevel::INFO_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define MLIR_PGX_ERROR(dialect, msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::ERROR_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            pgx::get_logger().log(pgx::LogLevel::ERROR_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

// Runtime-specific logging macros
#define RUNTIME_PGX_DEBUG(component, msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::DEBUG_LVL)) { \
            std::string formatted_msg = std::string("[RUNTIME-") + component + "] " + msg; \
            pgx::get_logger().log(pgx::LogLevel::DEBUG_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define RUNTIME_PGX_NOTICE(component, msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::INFO_LVL)) { \
            std::string formatted_msg = std::string("[RUNTIME-") + component + "] " + msg; \
            pgx::get_logger().log(pgx::LogLevel::INFO_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

} // namespace pgx