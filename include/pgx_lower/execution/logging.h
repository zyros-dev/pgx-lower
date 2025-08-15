#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

// Add PostgreSQL integration when building as extension
#ifdef POSTGRESQL_EXTENSION
// Push/pop PostgreSQL macros to avoid conflicts
// Save PostgreSQL macros that conflict with C++
#pragma push_macro("_")
#pragma push_macro("gettext")
#pragma push_macro("dgettext") 
#pragma push_macro("ngettext")
#pragma push_macro("dngettext")
#pragma push_macro("dcgettext")

// Undefine conflicting macros before including PostgreSQL headers
#undef _
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext
#undef dcgettext

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
        if (get_logger().should_log(LogLevel::TRACE_LVL)) \
            get_logger().log(LogLevel::TRACE_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_DEBUG(msg) \
    do { \
        if (get_logger().should_log(LogLevel::DEBUG_LVL)) \
            get_logger().log(LogLevel::DEBUG_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_INFO(msg) \
    do { \
        if (get_logger().should_log(LogLevel::INFO_LVL)) { \
            get_logger().log(LogLevel::INFO_LVL, __FILE__, __LINE__, msg); \
        } \
    } while (0)

#define PGX_WARNING(msg) \
    do { \
        if (get_logger().should_log(LogLevel::WARNING_LVL)) \
            get_logger().log(LogLevel::WARNING_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_ERROR(msg) \
    do { \
        if (get_logger().should_log(LogLevel::ERROR_LVL)) \
            get_logger().log(LogLevel::ERROR_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#ifdef POSTGRESQL_EXTENSION
#undef PGX_INFO
#define PGX_INFO(msg) \
    do { \
        elog(NOTICE, "[INFO] %s (%s:%d)", (std::string(msg)).c_str(), __FILE__, __LINE__); \
    } while (0)

#undef PGX_DEBUG
#define PGX_DEBUG(msg) \
    do { \
        elog(LOG, "[DEBUG] %s (%s:%d)", (std::string(msg)).c_str(), __FILE__, __LINE__); \
    } while (0)

#undef PGX_WARNING
#define PGX_WARNING(msg) \
    do { \
        elog(WARNING, "[WARNING] %s (%s:%d)", (std::string(msg)).c_str(), __FILE__, __LINE__); \
    } while (0)

#undef PGX_ERROR
#define PGX_ERROR(msg) \
    do { \
        elog(WARNING, "[ERROR] %s (%s:%d)", (std::string(msg)).c_str(), __FILE__, __LINE__); \
    } while (0)
#endif

#define PGX_NOTICE(msg) \
    do { \
        if (get_logger().should_log(LogLevel::INFO_LVL)) \
            get_logger().log(LogLevel::INFO_LVL, __FILE__, __LINE__, msg); \
    } while (0)

// MLIR-specific logging macros
#define MLIR_PGX_DEBUG(dialect, msg) \
    do { \
        if (get_logger().should_log(LogLevel::DEBUG_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            get_logger().log(LogLevel::DEBUG_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define MLIR_PGX_INFO(dialect, msg) \
    do { \
        if (get_logger().should_log(LogLevel::INFO_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            get_logger().log(LogLevel::INFO_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define MLIR_PGX_WARNING(dialect, msg) \
    do { \
        if (get_logger().should_log(LogLevel::WARNING_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            get_logger().log(LogLevel::WARNING_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define MLIR_PGX_ERROR(dialect, msg) \
    do { \
        if (get_logger().should_log(LogLevel::ERROR_LVL)) { \
            std::string formatted_msg = std::string("[") + dialect + "] " + msg; \
            get_logger().log(LogLevel::ERROR_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

// Runtime-specific logging macros
#define RUNTIME_PGX_DEBUG(component, msg) \
    do { \
        if (get_logger().should_log(LogLevel::DEBUG_LVL)) { \
            std::string formatted_msg = std::string("[RUNTIME-") + component + "] " + msg; \
            get_logger().log(LogLevel::DEBUG_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

#define RUNTIME_PGX_NOTICE(component, msg) \
    do { \
        if (get_logger().should_log(LogLevel::INFO_LVL)) { \
            std::string formatted_msg = std::string("[RUNTIME-") + component + "] " + msg; \
            get_logger().log(LogLevel::INFO_LVL, __FILE__, __LINE__, formatted_msg); \
        } \
    } while (0)

