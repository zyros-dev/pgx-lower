#ifndef EXECUTOR_C_H
#define EXECUTOR_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include <executor/execdesc.h>
#include "fmgr.h"

// Direct C++ function call
bool try_cpp_executor_direct(QueryDesc* queryDesc);

// PostgreSQL function interface
Datum try_cpp_executor(PG_FUNCTION_ARGS);
Datum log_cpp_notice(PG_FUNCTION_ARGS);

#ifdef __cplusplus
}
#endif

#endif
