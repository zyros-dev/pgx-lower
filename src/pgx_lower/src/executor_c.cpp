#include "executor_c.h"
#include "my_executor.h"
#include "postgres.h"
#include "fmgr.h"
#include "mlir/IR/MLIRContext.h"
#include <execinfo.h>
#include <exception>
#include <sstream>

extern "C" {

static void log_cpp_backtrace() {
    void *array[32];
    size_t size = backtrace(array, 32);
    char **strings = backtrace_symbols(array, size);
    if (strings) {
        std::ostringstream oss;
        oss << "C++ backtrace:" << std::endl;
        for (size_t i = 0; i < size; ++i) {
            oss << strings[i] << std::endl;
        }
        elog(LOG, "%s", oss.str().c_str());
        free(strings);
    }
}

bool try_cpp_executor_direct(QueryDesc* queryDesc) {
    try {
        // Test MLIR linking by creating a context
        mlir::MLIRContext context;
        elog(NOTICE, "Successfully created MLIR context!");
        return MyCppExecutor::execute(queryDesc);
    } catch (const std::exception& ex) {
        elog(ERROR, "C++ exception: %s", ex.what());
        log_cpp_backtrace();
        return false;
    } catch (...) {
        elog(ERROR, "Unknown C++ exception occurred!");
        log_cpp_backtrace();
        return false;
    }
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
