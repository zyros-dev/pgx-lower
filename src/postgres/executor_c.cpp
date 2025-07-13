// Include system and MLIR headers first to avoid libintl.h conflicts
#include <execinfo.h>
#include <exception>
#include <sstream>
#include "mlir/IR/MLIRContext.h"
#include "../../include/core/logging.h"

// Prevent gettext macro conflicts
#ifdef gettext
#undef gettext
#endif
#ifdef dgettext
#undef dgettext
#endif
#ifdef ngettext
#undef ngettext
#endif
#ifdef dngettext
#undef dngettext
#endif

#include "postgres/executor_c.h"
#include "postgres/my_executor.h"

extern "C" {

static void log_cpp_backtrace() {
    void* array[32];
    const size_t size = backtrace(array, 32);
    if (char** strings = backtrace_symbols(array, size)) {
        std::ostringstream oss;
        oss << "C++ backtrace:" << std::endl;
        for (size_t i = 0; i < size; ++i) {
            oss << strings[i] << std::endl;
        }
        elog(LOG, "%s", oss.str().c_str());
        free(strings);
    }
}

bool try_cpp_executor_direct(const QueryDesc* queryDesc) {
    try {
        // Test MLIR linking by creating a context
        mlir::MLIRContext context;
        PGX_INFO("Successfully created MLIR context!");

        // Create an instance of MyCppExecutor and call execute
        MyCppExecutor executor;
        return executor.execute(queryDesc);
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
Datum try_cpp_executor(PG_FUNCTION_ARGS) {
    const auto queryDesc = reinterpret_cast<QueryDesc*>(PG_GETARG_POINTER(0));
    const bool result = try_cpp_executor_direct(queryDesc);
    PG_RETURN_BOOL(result);
}

PG_FUNCTION_INFO_V1(log_cpp_notice);
Datum log_cpp_notice(PG_FUNCTION_ARGS) {
    elog(NOTICE, "Hello from C++!");
    PG_RETURN_VOID();
}

} // extern "C"
