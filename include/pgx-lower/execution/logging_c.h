#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper functions for PGX logging system
void pgx_debug_c(const char* message);
void pgx_info_c(const char* message);
void pgx_notice_c(const char* message);
void pgx_warning_c(const char* message);
void pgx_error_c(const char* message);

// C macros that use the wrapper functions
#define PGX_DEBUG_C(msg) pgx_debug_c(msg)
#define PGX_INFO_C(msg) pgx_info_c(msg)
#define PGX_NOTICE_C(msg) pgx_notice_c(msg)
#define PGX_WARNING_C(msg) pgx_warning_c(msg)
#define PGX_ERROR_C(msg) pgx_error_c(msg)

#ifdef __cplusplus
}
#endif