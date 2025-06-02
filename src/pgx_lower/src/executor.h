#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "postgres.h"
#include "executor/execdesc.h"
#include "executor/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare C-linkage functions here if needed
// Example:
// bool try_cpp_executor(QueryDesc* queryDesc);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

struct MyCppPlan
{
};

class MyCppExecutor
{
public:
  static bool execute(const QueryDesc* plan);
};

#endif // __cplusplus

#endif //EXECUTOR_H
