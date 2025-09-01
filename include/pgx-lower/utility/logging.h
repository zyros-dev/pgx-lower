#pragma once

#include <cstdarg>
#include <cstring>
#include <string>
#include <set>

#ifndef POSTGRESQL_EXTENSION
#include <iostream>
#endif

#ifdef POSTGRESQL_EXTENSION
// Push/pop PostgreSQL macros to avoid conflicts
#pragma push_macro("_")
#pragma push_macro("gettext")
#pragma push_macro("dgettext") 
#pragma push_macro("ngettext")
#pragma push_macro("dngettext")
#pragma push_macro("dcgettext")
#pragma push_macro("restrict")

#undef _
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext
#undef dcgettext
#undef restrict

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
#include "utils/guc.h"
}

#pragma pop_macro("restrict")
#pragma pop_macro("dcgettext")
#pragma pop_macro("dngettext")
#pragma pop_macro("ngettext")
#pragma pop_macro("dgettext")
#pragma pop_macro("gettext")
#pragma pop_macro("_")
#endif

namespace pgx_lower {
namespace log {

// Categories - WHAT is logging
enum class Category {
    AST_TRANSLATE,   // PostgreSQL AST → RelAlg
    RELALG_LOWER,    // RelAlg → DB+DSA
    DB_LOWER,        // DB → Standard  
    DSA_LOWER,       // DSA → Standard
    UTIL_LOWER,      // Util → LLVM
    RUNTIME,         // Runtime function calls
    JIT,             // JIT compilation/execution
    GENERAL          // General debug messages
};

// Levels - Different types of logs (independent switches)
enum class Level {
    IO,              // Input/Output boundaries
    DEBUG,           // General debug information
    TRACE            // Detailed trace information
};

extern bool log_enable;
extern bool log_io;
extern bool log_debug;
extern bool log_trace;
extern std::set<Category> enabled_categories;

void log(Category cat, Level level, const char* file, int line, const char* fmt, ...);

const char* category_name(Category cat);
const char* level_name(Level level);
bool should_log(Category cat, Level level);
const char* basename_only(const char* filepath);

} // namespace log
} // namespace pgx_lower

// Single macro with printf-style formatting and file/line info
#ifdef POSTGRESQL_EXTENSION
#define PGX_LOG(category, level, fmt, ...) \
    ::pgx_lower::log::log(::pgx_lower::log::Category::category, \
                          ::pgx_lower::log::Level::level, \
                          __FILE__, __LINE__, \
                          fmt, ##__VA_ARGS__)
#else
// For unit tests, just print to stdout/stderr with file:line
#define PGX_LOG(category, level, fmt, ...) \
    do { \
        const char* _basename = strrchr(__FILE__, '/'); \
        _basename = _basename ? _basename + 1 : __FILE__; \
        fprintf(stderr, "[%s:%s] %s:%d: " fmt "\n", \
                #category, #level, _basename, __LINE__, ##__VA_ARGS__); \
    } while(0)
#endif

// WARNING and ERROR always log - these are critical user-facing messages
#ifdef POSTGRESQL_EXTENSION
#define PGX_WARNING(fmt, ...) \
    elog(WARNING, fmt, ##__VA_ARGS__)

#define PGX_ERROR(fmt, ...) \
    elog(ERROR, fmt, ##__VA_ARGS__)
#else
#define PGX_WARNING(fmt, ...) \
    fprintf(stderr, "[WARNING] " fmt "\n", ##__VA_ARGS__)

#define PGX_ERROR(fmt, ...) \
    fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

