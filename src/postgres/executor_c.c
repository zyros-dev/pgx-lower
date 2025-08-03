#include "postgres/executor_c.h"
#include "core/logging_c.h"

#include "executor/execdesc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "postgres.h"
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

PG_MODULE_MAGIC;

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

static void segfault_handler(const int sig) {
    void *array[32];
    const size_t size = backtrace(array, 32);
    char **strings = backtrace_symbols(array, size);
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Caught signal %d (SIGSEGV) in extension!", sig);
    PGX_INFO_C(buffer);
    if (strings) {
        for (size_t i = 0; i < size; ++i) {
            char trace_buffer[512];
            snprintf(trace_buffer, sizeof(trace_buffer), "  %s", strings[i]);
            PGX_INFO_C(trace_buffer);
        }
        free(strings);
    }
    fflush(stderr);
    _exit(128 + sig);
}

void _PG_init(void) {
    PGX_NOTICE_C("Installing custom executor hook...");
    prev_ExecutorRun_hook = ExecutorRun_hook;
    ExecutorRun_hook = custom_executor;
    // Enable SIGSEGV handler for proper debugging
    PGX_NOTICE_C("SIGSEGV handler enabled for debugging!");
    signal(SIGSEGV, segfault_handler);
    
}

void _PG_fini(void) {
    PGX_NOTICE_C("Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}
