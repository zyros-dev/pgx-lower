#include "executor.h"

bool MyCppExecutor::execute(const QueryDesc* plan) {
  elog(NOTICE, "Custom executor is being executed in C++!");
  return false;
}
