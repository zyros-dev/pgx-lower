#ifndef EXECUTOR_C_H
#define EXECUTOR_C_H

#include "postgres.h"
#include "executor/executor.h"

#ifdef __cplusplus
extern "C" {
#endif


  bool try_cpp_executor(QueryDesc* queryDesc);

#ifdef __cplusplus
}
#endif

#endif
