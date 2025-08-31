#include "pgx-lower/utility/logging.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <algorithm>

namespace pgx_lower {
namespace log {

// GUC variables - initialized with defaults
bool log_enable = false;     // Master switch (off by default)
bool log_io = true;          // Show IO when enabled
bool log_debug = false;      // Don't show debug by default
bool log_trace = false;      // Don't show trace by default
std::set<Category> enabled_categories;  // Empty = all categories

static bool initialized = false;

static void initialize_if_needed() {
    if (initialized) return;
    initialized = true;
    
    // Check environment variables for initial configuration
    const char* enable_env = std::getenv("PGX_LOWER_LOG_ENABLE");
    if (enable_env && strcmp(enable_env, "true") == 0) {
        log_enable = true;
    }
    
    const char* io_env = std::getenv("PGX_LOWER_LOG_IO");
    if (io_env && strcmp(io_env, "false") == 0) {
        log_io = false;
    }
    
    const char* debug_env = std::getenv("PGX_LOWER_LOG_DEBUG");
    if (debug_env && strcmp(debug_env, "true") == 0) {
        log_debug = true;
    }
    
    const char* trace_env = std::getenv("PGX_LOWER_LOG_TRACE");
    if (trace_env && strcmp(trace_env, "true") == 0) {
        log_trace = true;
    }
    
    // Parse categories if specified
    const char* cat_env = std::getenv("PGX_LOWER_LOG_CATEGORIES");
    if (cat_env && strlen(cat_env) > 0) {
        std::string cats(cat_env);
        std::stringstream ss(cats);
        std::string cat;
        while (std::getline(ss, cat, ',')) {
            cat.erase(0, cat.find_first_not_of(" \t"));
            cat.erase(cat.find_last_not_of(" \t") + 1);
            
            // Convert to lowercase for comparison
            std::transform(cat.begin(), cat.end(), cat.begin(), ::tolower);
            
            if (cat == "ast_translate") enabled_categories.insert(Category::AST_TRANSLATE);
            else if (cat == "relalg_lower") enabled_categories.insert(Category::RELALG_LOWER);
            else if (cat == "db_lower") enabled_categories.insert(Category::DB_LOWER);
            else if (cat == "dsa_lower") enabled_categories.insert(Category::DSA_LOWER);
            else if (cat == "util_lower") enabled_categories.insert(Category::UTIL_LOWER);
            else if (cat == "runtime") enabled_categories.insert(Category::RUNTIME);
            else if (cat == "jit") enabled_categories.insert(Category::JIT);
            else if (cat == "general") enabled_categories.insert(Category::GENERAL);
        }
    }
}

const char* category_name(Category cat) {
    switch(cat) {
        case Category::AST_TRANSLATE: return "AST_TRANSLATE";
        case Category::RELALG_LOWER: return "RELALG_LOWER";
        case Category::DB_LOWER: return "DB_LOWER";
        case Category::DSA_LOWER: return "DSA_LOWER";
        case Category::UTIL_LOWER: return "UTIL_LOWER";
        case Category::RUNTIME: return "RUNTIME";
        case Category::JIT: return "JIT";
        case Category::GENERAL: return "GENERAL";
    }
    return "UNKNOWN";
}

const char* level_name(Level level) {
    switch(level) {
        case Level::IO: return "IO";
        case Level::DEBUG: return "DEBUG";
        case Level::TRACE: return "TRACE";
    }
    return "UNKNOWN";
}

bool should_log(Category cat, Level level) {
    initialize_if_needed();
    
    // Master switch
    if (!log_enable) return false;
    
    // Check level
    switch(level) {
        case Level::IO:
            if (!log_io) return false;
            break;
        case Level::DEBUG:
            if (!log_debug) return false;
            break;
        case Level::TRACE:
            if (!log_trace) return false;
            break;
    }
    
    // Check category filter (empty means all categories)
    if (!enabled_categories.empty()) {
        if (enabled_categories.find(cat) == enabled_categories.end()) {
            return false;
        }
    }
    
    return true;
}

void log(Category cat, Level level, const char* fmt, ...) {
    // Check if we should log this
    if (!should_log(cat, level)) return;
    
    // Format the message
    char message[8192];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    
#ifdef POSTGRESQL_EXTENSION
    // Output using PostgreSQL's elog at DEBUG1 level
    // This requires client_min_messages = DEBUG1 to be visible
    elog(DEBUG1, "[%s:%s] %s", category_name(cat), level_name(level), message);
#else
    // For unit tests, output to stderr
    fprintf(stderr, "[%s:%s] %s\n", category_name(cat), level_name(level), message);
#endif
}

} // namespace log
} // namespace pgx_lower

extern "C" void pgx_update_log_settings(bool enable, bool debug, bool io, bool trace, const char* categories) {
    using namespace pgx_lower::log;
    
    log_enable = enable;
    log_debug = debug;
    log_io = io;
    log_trace = trace;
    
    // Parse categories
    enabled_categories.clear();
    if (categories && strlen(categories) > 0) {
        std::string cats(categories);
        std::stringstream ss(cats);
        std::string cat;
        while (std::getline(ss, cat, ',')) {
            cat.erase(0, cat.find_first_not_of(" \t"));
            cat.erase(cat.find_last_not_of(" \t") + 1);
            
            // Convert to lowercase for comparison
            std::transform(cat.begin(), cat.end(), cat.begin(), ::tolower);
            
            if (cat == "ast_translate") enabled_categories.insert(Category::AST_TRANSLATE);
            else if (cat == "relalg_lower") enabled_categories.insert(Category::RELALG_LOWER);
            else if (cat == "db_lower") enabled_categories.insert(Category::DB_LOWER);
            else if (cat == "dsa_lower") enabled_categories.insert(Category::DSA_LOWER);
            else if (cat == "util_lower") enabled_categories.insert(Category::UTIL_LOWER);
            else if (cat == "runtime") enabled_categories.insert(Category::RUNTIME);
            else if (cat == "jit") enabled_categories.insert(Category::JIT);
            else if (cat == "general") enabled_categories.insert(Category::GENERAL);
        }
    }
}