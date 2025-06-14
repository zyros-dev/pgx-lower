#include "executor_c.h"
#include "executor.h"
#include "postgres.h"
#include "fmgr.h"

extern "C" {

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(try_cpp_executor);
Datum
try_cpp_executor(PG_FUNCTION_ARGS)
{
    QueryDesc* queryDesc = (QueryDesc*)PG_GETARG_POINTER(0);
    bool result = MyCppExecutor::execute(queryDesc);
    PG_RETURN_BOOL(result);
}

PG_FUNCTION_INFO_V1(log_cpp_notice);
Datum
log_cpp_notice(PG_FUNCTION_ARGS)
{
    elog(NOTICE, "Hello from C++!");
    PG_RETURN_VOID();
}

} // extern "C" 