#include "pgx-lower/utility/logging_c.h"
#include "pgx-lower/utility/logging.h"

extern "C" {

void pgx_debug_c(const char* message) {
    PGX_LOG(GENERAL, DEBUG, "%s", message);
}

void pgx_info_c(const char* message) {
    PGX_LOG(GENERAL, DEBUG, "%s", message);
}

void pgx_notice_c(const char* message) {
    PGX_LOG(GENERAL, DEBUG, "%s", message);
}

void pgx_warning_c(const char* message) {
    PGX_WARNING("%s", message);
}

void pgx_error_c(const char* message) {
    PGX_ERROR("%s", message);
}

}