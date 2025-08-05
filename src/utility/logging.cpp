#include "execution/logging.h"
#include <cstring>

namespace pgx {

Logger::Logger()
: current_level(LogLevel::INFO_LVL)
, debug_to_file(false) {
    // Read environment variables
    const char* log_level_env = std::getenv("PGX_LOWER_LOG_LEVEL");
    if (log_level_env) {
        if (strcmp(log_level_env, "DEBUG") == 0) {
            current_level = LogLevel::DEBUG_LVL;
        }
        else if (strcmp(log_level_env, "INFO") == 0) {
            current_level = LogLevel::INFO_LVL;
        }
        else if (strcmp(log_level_env, "WARNING") == 0) {
            current_level = LogLevel::WARNING_LVL;
        }
        else if (strcmp(log_level_env, "ERROR") == 0) {
            current_level = LogLevel::ERROR_LVL;
        }
    }

    const char* debug_file_env = std::getenv("PGX_LOWER_DEBUG_FILE");
    if (debug_file_env) {
        set_debug_file(debug_file_env);
    }
}

Logger::~Logger() {
    if (debug_file.is_open()) {
        debug_file.close();
    }
}

void Logger::set_level(LogLevel level) {
    current_level = level;
}

void Logger::set_debug_file(const std::string& filename) {
    debug_file.open(filename, std::ios::app);
    if (debug_file.is_open()) {
        debug_to_file = true;
    }
}

bool Logger::should_log(LogLevel level) const {
    return static_cast<int>(level) >= static_cast<int>(current_level);
}

void Logger::log(LogLevel level, const char* file, int line, const std::string& message) {
    if (!should_log(level)) {
        return;
    }

    const char* level_str = "";
    switch (level) {
    case LogLevel::DEBUG_LVL: level_str = "DEBUG"; break;
    case LogLevel::INFO_LVL: level_str = "INFO"; break;
    case LogLevel::WARNING_LVL: level_str = "WARNING"; break;
    case LogLevel::ERROR_LVL: level_str = "ERROR"; break;
    }

    // Extract just the filename from the full path
    const char* filename = strrchr(file, '/');
    if (filename) {
        filename++; // Skip the '/'
    }
    else {
        filename = file;
    }

    const std::string formatted_message = std::string("[") + level_str + "] " + message;

    // Use unified stream output - the PGX macros in the header handle PostgreSQL integration
    if (level == LogLevel::DEBUG_LVL && debug_to_file && debug_file.is_open()) {
        // Send DEBUG logs to file if configured
        debug_file << formatted_message << " (" << filename << ":" << line << ")" << std::endl;
        debug_file.flush();
    }
    else if (level == LogLevel::DEBUG_LVL && !debug_to_file) {
        // Send DEBUG logs to stderr by default
        std::cerr << formatted_message << std::endl;
    }
    else {
        // Send INFO/WARNING/ERROR to stdout for regression tests
        std::cout << formatted_message << std::endl;
    }
}

Logger& get_logger() {
    static Logger instance;
    return instance;
}

} // namespace pgx