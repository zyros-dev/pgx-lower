#include "executor_c.h"

#include "executor/execdesc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "postgres.h"
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

PG_MODULE_MAGIC;

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL; // NOLINT(*-avoid-non-const-global-variables)

static bool try_cpp_executor_internal(QueryDesc *queryDesc) {
    elog(NOTICE, "Calling C++ executor from C...");
    return try_cpp_executor_direct(queryDesc);
}

static void custom_executor(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once) {
    elog(NOTICE, "Custom executor is being executed in C!");

    try_cpp_executor_internal(queryDesc);

    if (prev_ExecutorRun_hook) {
        prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
    }
    else {
        standard_ExecutorRun(queryDesc, direction, count, execute_once);
    }
}

static void segfault_handler(int sig) {
    void *array[32];
    size_t size = backtrace(array, 32);
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
    // Register SIGSEGV handler for debugging
    elog(NOTICE, "Registering custom sigsegv handler!");
    signal(SIGSEGV, segfault_handler);
}

void _PG_fini(void) {
    elog(NOTICE, "Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}
