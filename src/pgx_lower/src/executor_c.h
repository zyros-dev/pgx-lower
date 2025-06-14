#ifndef EXECUTOR_C_H
#define EXECUTOR_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fmgr.h"
#include "postgres.h"

Datum try_cpp_executor(PG_FUNCTION_ARGS);
Datum log_cpp_notice(PG_FUNCTION_ARGS);

#ifdef __cplusplus
}
#endif

#endif
