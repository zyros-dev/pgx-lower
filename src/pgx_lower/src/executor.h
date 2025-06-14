#ifndef EXECUTOR_H
#define EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

struct QueryDesc;

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
