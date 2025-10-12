#pragma once

#include <cstdarg>
#include <cstring>
#include <string>
#include <set>
#include <sstream>
#include <execinfo.h>
#include <cxxabi.h>
#include <memory>

#ifndef POSTGRESQL_EXTENSION
#include <iostream>
#endif

namespace mlir {
    class Value;
    class Type;
}

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

enum class Category {
    AST_TRANSLATE,
    RELALG_LOWER,
    DB_LOWER,
    DSA_LOWER,
    UTIL_LOWER,
    RUNTIME,
    JIT,
    GENERAL,
    PROBLEM
};

enum class Level {
    IO,
    DEBUG,
    IR,
    TRACE,
    WARNING_LEVEL,
    ERROR_LEVEL,
};

extern bool log_enable;
extern bool log_io;
extern bool log_debug;
extern bool log_ir;
extern bool log_trace;
extern std::set<Category> enabled_categories;

void log(Category cat, Level level, const char* file, int line, const char* fmt, ...);

const char* category_name(Category cat);
const char* level_name(Level level);
bool should_log(Category cat, Level level);
const char* basename_only(const char* filepath);

inline auto capture_stacktrace(int skip_frames = 1, int max_frames = 20) -> std::string {
    void* buffer[max_frames];
    const int n_frames = backtrace(buffer, max_frames);

    if (n_frames <= skip_frames)
        return "failed to get stacktrace";

    std::unique_ptr<char*, decltype(&free)> symbols(
        backtrace_symbols(buffer, n_frames),
        &free
    );

    if (!symbols) {
        return "failed to get stacktrace";
    }

    auto oss = std::ostringstream{};
    oss << "\nStack trace:";

    for (int i = skip_frames; i < n_frames; ++i) {
        std::string symbol_str(symbols.get()[i]);
        size_t start_paren = symbol_str.find('(');
        size_t plus_sign = symbol_str.find('+', start_paren);

        if (start_paren != std::string::npos && plus_sign != std::string::npos) {
            std::string mangled_name = symbol_str.substr(start_paren + 1, plus_sign - start_paren - 1);

            int status = 0;
            std::unique_ptr<char, decltype(&free)> demangled(
                abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status),
                &free
            );

            if (status == 0 && demangled) {
                symbol_str = symbol_str.substr(0, start_paren + 1) +
                            std::string(demangled.get()) +
                            symbol_str.substr(plus_sign);
            }
        }

        oss << "\n  #" << (i - skip_frames) << " " << symbol_str;
    }

    return oss.str();
}

// MLIR value verification and printing
auto verify_and_print(const mlir::Value val) -> void;

auto print_type(const mlir::Type val) -> void;
auto type_to_string(const mlir::Type type) -> std::string;
auto value_to_string(const mlir::Value val) -> std::string;

class ScopeLogger {
public:
    ScopeLogger(Category cat, const char* file, int line, const char* function_name);
    ~ScopeLogger();

    ScopeLogger(const ScopeLogger&) = delete;
    ScopeLogger& operator=(const ScopeLogger&) = delete;
    ScopeLogger(ScopeLogger&&) = delete;
    ScopeLogger& operator=(ScopeLogger&&) = delete;

private:
    Category category_;
    const char* file_;
    int line_;
    std::string function_name_;
};

} // namespace log
} // namespace pgx_lower

#ifdef POSTGRESQL_EXTENSION
#define PGX_LOG(category, level, fmt, ...) \
    ::pgx_lower::log::log(::pgx_lower::log::Category::category, \
                          ::pgx_lower::log::Level::level, \
                          __FILE__, __LINE__, \
                          fmt, ##__VA_ARGS__)
#else
#define PGX_LOG(category, level, fmt, ...) \
    do { \
        const char* _basename = strrchr(__FILE__, '/'); \
        _basename = _basename ? _basename + 1 : __FILE__; \
        fprintf(stderr, "[%s:%s] %s:%d: " fmt "\n", \
                #category, #level, _basename, __LINE__, ##__VA_ARGS__); \
    } while(0)
#endif

#ifdef POSTGRESQL_EXTENSION
#define PGX_WARNING(fmt, ...) \
    ::pgx_lower::log::log(::pgx_lower::log::Category::PROBLEM, \
                          ::pgx_lower::log::Level::WARNING_LEVEL, \
                          __FILE__, __LINE__, \
                          fmt, ##__VA_ARGS__)
#define PGX_ERROR(fmt, ...) \
    do { \
        auto _pgx_stacktrace = ::pgx_lower::log::capture_stacktrace(2); \
        ::pgx_lower::log::log(::pgx_lower::log::Category::PROBLEM, \
                              ::pgx_lower::log::Level::ERROR_LEVEL, \
                              __FILE__, __LINE__, \
                              fmt "%s", ##__VA_ARGS__, _pgx_stacktrace.c_str()); \
    } while(0)
#else
#define PGX_WARNING(fmt, ...) \
    fprintf(stderr, "[WARNING] " fmt "\n", ##__VA_ARGS__)

#define PGX_ERROR(fmt, ...) \
    do { \
        auto _pgx_stacktrace = ::pgx_lower::log::capture_stacktrace(2); \
        fprintf(stderr, "[ERROR] " fmt "%s\n", ##__VA_ARGS__, _pgx_stacktrace.c_str()); \
    } while(0)
#endif

#define PGX_IO(category) \
    ::pgx_lower::log::ScopeLogger _pgx_io_logger( \
        ::pgx_lower::log::Category::category, \
        __FILE__, __LINE__, \
        __func__)

