#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

namespace pgx {

enum class LogLevel {
    DEBUG_LVL = 0,
    INFO_LVL = 1,
    WARNING_LVL = 2,
    ERROR_LVL = 3
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
#define PGX_DEBUG(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::DEBUG_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::DEBUG_LVL, __FILE__, __LINE__, msg); \
    } while (0)

#define PGX_INFO(msg) \
    do { \
        if (pgx::get_logger().should_log(pgx::LogLevel::INFO_LVL)) \
            pgx::get_logger().log(pgx::LogLevel::INFO_LVL, __FILE__, __LINE__, msg); \
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

} // namespace pgx