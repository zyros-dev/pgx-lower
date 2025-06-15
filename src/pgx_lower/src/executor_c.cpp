#include "executor_c.h"
#include "my_executor.h"
#include "postgres.h"
#include "fmgr.h"
#include "mlir/IR/MLIRContext.h"

extern "C" {

bool try_cpp_executor_direct(QueryDesc* queryDesc) {
    // Test MLIR linking by creating a context
    mlir::MLIRContext context;
    elog(NOTICE, "Successfully created MLIR context!");

    return MyCppExecutor::execute(queryDesc);
}

PG_FUNCTION_INFO_V1(try_cpp_executor);
Datum
try_cpp_executor(PG_FUNCTION_ARGS)
{
    QueryDesc* queryDesc = (QueryDesc*)PG_GETARG_POINTER(0);
    bool result = try_cpp_executor_direct(queryDesc);
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
