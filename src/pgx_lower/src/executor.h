#ifndef EXECUTOR_H
#define EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "nodes/execnodes.h"
#include <executor/execdesc.h>

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

struct MyCppPlan {};

class MyCppExecutor {
   public:
    static bool execute(const QueryDesc* plan);
};

#endif  // __cplusplus

#endif  // EXECUTOR_H
