#include "execution/postgres/executor_c.h"
#include "execution/logging_c.h"

#include "executor/execdesc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "postgres.h"
// Removed signal.h, execinfo.h, unistd.h - not needed without custom signal handler

PG_MODULE_MAGIC;

// External reference to the global flag defined in executor_c.cpp
extern bool g_extension_after_load;

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL; // NOLINT(*-avoid-non-const-global-variables)


static bool try_cpp_executor_internal(const QueryDesc *queryDesc) {
    PGX_NOTICE_C("Calling C++ executor from C...");
    return try_cpp_executor_direct(queryDesc);
}

static void custom_executor(QueryDesc *queryDesc, const ScanDirection direction, const uint64 count, const bool execute_once) {
    PGX_NOTICE_C("Custom executor is being executed in C!");

    const bool mlir_handled = try_cpp_executor_internal(queryDesc);

    if (!mlir_handled) {
        PGX_NOTICE_C("MLIR couldn't handle query, falling back to standard executor");
        if (prev_ExecutorRun_hook) {
            prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
        }
        else {
            standard_ExecutorRun(queryDesc, direction, count, execute_once);
        }
    }
    else {
        PGX_NOTICE_C("MLIR successfully handled the query");
    }
    
}

void _PG_init(void) {
    PGX_NOTICE_C("Installing custom executor hook...");
    prev_ExecutorRun_hook = ExecutorRun_hook;
    ExecutorRun_hook = custom_executor;
    
    // CRITICAL FIX: Set LOAD detection flag
    g_extension_after_load = true;
    PGX_NOTICE_C("LOAD detection flag set - memory context protection enabled");
    
    // Initialize MLIR pass registration
    // This needs to be done once at startup to ensure passes can be created
    PGX_NOTICE_C("Initializing MLIR pass registration...");
    extern void initialize_mlir_passes(void);
    initialize_mlir_passes();
}

void _PG_fini(void) {
    PGX_NOTICE_C("Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}
