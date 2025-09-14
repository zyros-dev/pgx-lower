#include "pgx-lower/utility/logging.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <vector>

namespace pgx_lower { namespace log {

bool log_enable = false;
bool log_io = true;
bool log_debug = false;
bool log_ir = false;
bool log_trace = false;
std::set<Category> enabled_categories;

static bool initialized = false;

static void initialize_if_needed() {
    if (initialized)
        return;
    initialized = true;

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

    const char* ir_env = std::getenv("PGX_LOWER_LOG_IR");
    if (ir_env && strcmp(ir_env, "true") == 0) {
        log_ir = true;
    }

    const char* trace_env = std::getenv("PGX_LOWER_LOG_TRACE");
    if (trace_env && strcmp(trace_env, "true") == 0) {
        log_trace = true;
    }

    const char* cat_env = std::getenv("PGX_LOWER_LOG_CATEGORIES");
    if (cat_env && strlen(cat_env) > 0) {
        std::string cats(cat_env);
        std::stringstream ss(cats);
        std::string cat;
        while (std::getline(ss, cat, ',')) {
            cat.erase(0, cat.find_first_not_of(" \t"));
            cat.erase(cat.find_last_not_of(" \t") + 1);

            std::transform(cat.begin(), cat.end(), cat.begin(), ::tolower);

            if (cat == "ast_translate")
                enabled_categories.insert(Category::AST_TRANSLATE);
            else if (cat == "relalg_lower")
                enabled_categories.insert(Category::RELALG_LOWER);
            else if (cat == "db_lower")
                enabled_categories.insert(Category::DB_LOWER);
            else if (cat == "dsa_lower")
                enabled_categories.insert(Category::DSA_LOWER);
            else if (cat == "util_lower")
                enabled_categories.insert(Category::UTIL_LOWER);
            else if (cat == "runtime")
                enabled_categories.insert(Category::RUNTIME);
            else if (cat == "jit")
                enabled_categories.insert(Category::JIT);
            else if (cat == "general")
                enabled_categories.insert(Category::GENERAL);
        }
    }
}

const char* category_name(Category cat) {
    switch (cat) {
    case Category::AST_TRANSLATE: return "AST_TRANSLATE";
    case Category::RELALG_LOWER: return "RELALG_LOWER";
    case Category::DB_LOWER: return "DB_LOWER";
    case Category::DSA_LOWER: return "DSA_LOWER";
    case Category::UTIL_LOWER: return "UTIL_LOWER";
    case Category::RUNTIME: return "RUNTIME";
    case Category::JIT: return "JIT";
    case Category::GENERAL: return "GENERAL";
    case Category::PROBLEM: return "PROBLEM";
    }
    return "UNKNOWN";
}

const char* level_name(Level level) {
    switch (level) {
    case Level::IO: return "IO";
    case Level::DEBUG: return "DEBUG";
    case Level::IR: return "IR";
    case Level::TRACE: return "TRACE";
    case Level::WARNING_LEVEL: return "WARNING_LEVEL";
    case Level::ERROR_LEVEL: return "ERROR_LEVEL";
    }
    return "UNKNOWN";
}

bool should_log(Category cat, Level level) {
    initialize_if_needed();
    if (cat == Category::PROBLEM) return true;

    if (!log_enable)
        return false;

    switch (level) {
    case Level::IO:
        if (!log_io)
            return false;
        break;
    case Level::DEBUG:
        if (!log_debug)
            return false;
        break;
    case Level::IR:
        if (!log_ir)
            return false;
        break;
    case Level::TRACE:
        if (!log_trace)
            return false;
        break;
    case Level::WARNING_LEVEL:
    case Level::ERROR_LEVEL:
        // Always log warnings and errors when logging is enabled
        break;
    }

    if (!enabled_categories.empty()) {
        if (!enabled_categories.contains(cat)) {
            return false;
        }
    }

    return true;
}

const char* basename_only(const char* filepath) {
    const char* basename = strrchr(filepath, '/');
    return basename ? basename + 1 : filepath;
}

void log(Category cat, Level level, const char* file, int line, const char* fmt, ...) {
    if (!should_log(cat, level))
        return;

    // Adjust the buffer size per message so that we can log really long things
    va_list args_size;
    va_start(args_size, fmt);
    int size_needed = vsnprintf(nullptr, 0, fmt, args_size) + 1;
    va_end(args_size);

    std::vector<char> message(size_needed * 2);
    
    va_list args;
    va_start(args, fmt);
    vsnprintf(message.data(), size_needed, fmt, args);
    va_end(args);

    const char* filename = basename_only(file);

#ifdef POSTGRESQL_EXTENSION
    if (cat == Category::PROBLEM) {
        elog(WARNING, "[%s:%s] %s:%d: %s", category_name(cat), level_name(level), filename, line, message.data());
    } else {
        elog(DEBUG1, "[%s:%s] %s:%d: %s", category_name(cat), level_name(level), filename, line, message.data());
    }
#else
    // For unit tests, output to stderr
    fprintf(stderr, "[%s:%s] %s:%d: %s\n", category_name(cat), level_name(level), filename, line, message.data());
#endif
}

ScopeLogger::ScopeLogger(Category cat, const char* file, int line, const char* function_name)
    : category_(cat)
    , file_(file)
    , line_(line)
    , function_name_(function_name) {
    log(category_, Level::IO, file_, line_, "%s IN", function_name);
}

ScopeLogger::~ScopeLogger() {
    log(category_, Level::IO, file_, line_, "%s OUT", function_name_.c_str());
}

}} // namespace pgx_lower::log

extern "C" void pgx_update_log_settings(bool enable, bool debug, bool ir, bool io, bool trace, const char* categories) {
    using namespace pgx_lower::log;

    log_enable = enable;
    log_debug = debug;
    log_ir = ir;
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

            if (cat == "ast_translate")
                enabled_categories.insert(Category::AST_TRANSLATE);
            else if (cat == "relalg_lower")
                enabled_categories.insert(Category::RELALG_LOWER);
            else if (cat == "db_lower")
                enabled_categories.insert(Category::DB_LOWER);
            else if (cat == "dsa_lower")
                enabled_categories.insert(Category::DSA_LOWER);
            else if (cat == "util_lower")
                enabled_categories.insert(Category::UTIL_LOWER);
            else if (cat == "runtime")
                enabled_categories.insert(Category::RUNTIME);
            else if (cat == "jit")
                enabled_categories.insert(Category::JIT);
            else if (cat == "general")
                enabled_categories.insert(Category::GENERAL);
        }
    }
}