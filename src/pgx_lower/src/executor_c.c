#include "executor_c.h"

#include "executor/executor.h"
#include "fmgr.h"
#include "postgres.h"

PG_MODULE_MAGIC;

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL;

static bool try_cpp_executor_internal(QueryDesc *queryDesc) {
    // Placeholder for C++ call
    return false;
}

static void log_cpp_notice_internal(void) {
    elog(NOTICE, "Custom executor is being executed in C++!");
}

static void custom_executor(QueryDesc *queryDesc, ScanDirection direction,
                            uint64 count, bool execute_once) {
    elog(NOTICE, "Custom executor is being executed in C!");

    if (try_cpp_executor_internal(queryDesc))
        return;

    if (prev_ExecutorRun_hook)
        prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
    else
        standard_ExecutorRun(queryDesc, direction, count, execute_once);
}

void _PG_init(void) {
    elog(NOTICE, "Installing custom executor hook...");
    prev_ExecutorRun_hook = ExecutorRun_hook;
    ExecutorRun_hook = custom_executor;
}

void _PG_fini(void) {
    elog(NOTICE, "Uninstalling custom executor hook...");
    ExecutorRun_hook = NULL;
}

Datum log_cpp_notice(PG_FUNCTION_ARGS) {
    log_cpp_notice_internal();
    PG_RETURN_VOID();
}

Datum try_cpp_executor(PG_FUNCTION_ARGS) {
    QueryDesc *queryDesc = (QueryDesc *)PG_GETARG_POINTER(0);
    bool result = try_cpp_executor_internal(queryDesc);
    PG_RETURN_BOOL(result);
}
