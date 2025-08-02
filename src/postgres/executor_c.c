#include "postgres/executor_c.h"

#include "executor/execdesc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "postgres.h"
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

PG_MODULE_MAGIC;

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL; // NOLINT(*-avoid-non-const-global-variables)

// External reference to global flag defined in executor_c.cpp
extern bool g_extension_after_load;


static bool try_cpp_executor_internal(const QueryDesc *queryDesc) {
    elog(NOTICE, "Calling C++ executor from C...");
    return try_cpp_executor_direct(queryDesc);
}

static void custom_executor(QueryDesc *queryDesc, const ScanDirection direction, const uint64 count, const bool execute_once) {
    elog(NOTICE, "Custom executor is being executed in C!");

    const bool mlir_handled = try_cpp_executor_internal(queryDesc);

    if (!mlir_handled) {
        elog(NOTICE, "MLIR couldn't handle query, falling back to standard executor");
        if (prev_ExecutorRun_hook) {
            prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
        }
        else {
            standard_ExecutorRun(queryDesc, direction, count, execute_once);
        }
    }
    else {
        elog(NOTICE, "MLIR successfully handled the query");
    }
    
    // Reset the after_load flag after the first query executes
    // This allows subsequent queries to process expressions normally
    if (g_extension_after_load) {
        elog(NOTICE, "Resetting g_extension_after_load flag after first query");
        g_extension_after_load = false;
    }
}

static void segfault_handler(const int sig) {
    void *array[32];
    const size_t size = backtrace(array, 32);
    char **strings = backtrace_symbols(array, size);
    elog(LOG, "Caught signal %d (SIGSEGV) in extension!", sig);
    if (strings) {
        for (size_t i = 0; i < size; ++i) {
            elog(LOG, "  %s", strings[i]);
        }
        free(strings);
    }
    fflush(stderr);
    _exit(128 + sig);
}

void _PG_init(void) {
    elog(NOTICE, "Installing custom executor hook...");
    prev_ExecutorRun_hook = ExecutorRun_hook;
    ExecutorRun_hook = custom_executor;
    // Enable SIGSEGV handler for proper debugging
    elog(NOTICE, "SIGSEGV handler enabled for debugging!");
    signal(SIGSEGV, segfault_handler);
    
    // Mark that extension has been loaded - this recreates MLIR context
    g_extension_after_load = true;
}

void _PG_fini(void) {
    elog(NOTICE, "Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}
