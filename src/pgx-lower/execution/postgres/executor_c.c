#include "pgx-lower/execution/postgres/executor_c.h"
#include "pgx-lower/utility/logging_c.h"

#include "executor/execdesc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "postgres.h"
#include "utils/guc.h"

PG_MODULE_MAGIC;

extern bool g_extension_after_load;

static bool pgx_lower_log_enable = false;
static bool pgx_lower_log_debug = false;
static bool pgx_lower_log_ir = false;
static bool pgx_lower_log_io = false;
static bool pgx_lower_log_trace = false;
static char *pgx_lower_enabled_categories = NULL;

extern void pgx_update_log_settings(bool enable, bool debug, bool ir, bool io, bool trace, const char *categories);

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL;

static bool try_cpp_executor_internal(const QueryDesc *queryDesc) {
    PGX_NOTICE_C("Calling C++ executor from C...");
    return try_cpp_executor_direct(queryDesc);
}

static void
custom_executor(QueryDesc *queryDesc, const ScanDirection direction, const uint64 count, const bool execute_once) {
    PGX_NOTICE_C("Custom executor is being executed in C!");

    const bool mlir_handled = try_cpp_executor_internal(queryDesc);

    if (!mlir_handled) {
        if (prev_ExecutorRun_hook) {
            prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
        }
        else {
            standard_ExecutorRun(queryDesc, direction, count, execute_once);
        }
    }
}

static void update_logging(bool newval, void *extra) {
    pgx_update_log_settings(pgx_lower_log_enable, pgx_lower_log_debug, pgx_lower_log_ir, pgx_lower_log_io,
                           pgx_lower_log_trace, pgx_lower_enabled_categories);
}

static void register_bool_guc(const char *name, const char *desc, bool *var) {
    DefineCustomBoolVariable(name, desc, NULL, var, false, PGC_USERSET, 0, NULL, update_logging, NULL);
}

static void register_string_guc(const char *name, const char *desc, char **var) {
    DefineCustomStringVariable(name, desc, NULL, var, "", PGC_USERSET, 0, NULL, 
                              (GucStringAssignHook)update_logging, NULL);
}

void _PG_init(void) {
    PGX_NOTICE_C("Installing custom executor hook...");
    prev_ExecutorRun_hook = ExecutorRun_hook;
    ExecutorRun_hook = custom_executor;

    g_extension_after_load = true;
    PGX_NOTICE_C("LOAD detection flag set - memory context protection enabled");

    register_bool_guc("pgx_lower.log_enable", "Enable PGX-Lower logging", &pgx_lower_log_enable);
    register_bool_guc("pgx_lower.log_debug", "Enable debug logging", &pgx_lower_log_debug);
    register_bool_guc("pgx_lower.log_ir", "Enable IR (intermediate representation) logging", &pgx_lower_log_ir);
    register_bool_guc("pgx_lower.log_io", "Enable I/O boundary logging", &pgx_lower_log_io);
    register_bool_guc("pgx_lower.log_trace", "Enable trace logging", &pgx_lower_log_trace);
    register_string_guc("pgx_lower.enabled_categories", "Comma-separated list of enabled log categories", 
                       &pgx_lower_enabled_categories);

    PGX_NOTICE_C("Initializing MLIR pass registration...");
    extern void initialize_mlir_passes(void);
    initialize_mlir_passes();
}

void _PG_fini(void) {
    PGX_NOTICE_C("Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}
