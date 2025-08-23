#include "pgx-lower/execution/logging_c.h"
#include "pgx-lower/execution/logging.h"

extern "C" {

void pgx_debug_c(const char* message) {
    PGX_DEBUG(std::string(message));
}

void pgx_info_c(const char* message) {
    PGX_INFO(std::string(message));
}

void pgx_notice_c(const char* message) {
    PGX_NOTICE(std::string(message));
}

void pgx_warning_c(const char* message) {
    PGX_WARNING(std::string(message));
}

void pgx_error_c(const char* message) {
    PGX_ERROR(std::string(message));
}

}