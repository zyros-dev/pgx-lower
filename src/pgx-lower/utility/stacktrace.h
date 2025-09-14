#pragma once

#include <string>
#include <sstream>
#include <execinfo.h>
#include <cxxabi.h>
#include <memory>

namespace pgx_lower::utility {

inline auto capture_stacktrace(int skip_frames = 1, int max_frames = 20) -> std::string {
    void* buffer[max_frames];
    const int n_frames = backtrace(buffer, max_frames);

    if (n_frames <= skip_frames)
        return "No stack trace available";

    std::unique_ptr<char*, decltype(&free)> symbols(
        backtrace_symbols(buffer, n_frames),
        &free
    );

    if (!symbols) {
        return "Failed to get stack symbols";
    }

    auto oss = std::ostringstream{};
    oss << "Stack trace:\n";

    for (int i = skip_frames; i < n_frames; ++i) {
        std::string symbol_str(symbols.get()[i]);

        // Try to demangle C++ names
        // Format is typically: executable(symbol+offset) [address]
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
                // Replace mangled name with demangled one
                symbol_str = symbol_str.substr(0, start_paren + 1) +
                            std::string(demangled.get()) +
                            symbol_str.substr(plus_sign);
            }
        }

        oss << "  #" << (i - skip_frames) << " " << symbol_str << "\n";
    }

    return oss.str();
}

// Macro to include stacktrace in PGX_ERROR
#define PGX_ERROR_WITH_TRACE(fmt, ...) \
    do { \
        auto stacktrace = pgx_lower::utility::capture_stacktrace(); \
        PGX_ERROR(fmt "\n%s", ##__VA_ARGS__, stacktrace.c_str()); \
    } while(0)

} // namespace pgx_lower::utility