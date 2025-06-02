#include "postgres.h"
#include "executor/executor.h"
#include "fmgr.h"

#include "executor_c.h"

PG_MODULE_MAGIC;

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL;


static void custom_executor(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once)
{
  elog(NOTICE, "Custom executor is being executed in C!");

  if (try_cpp_executor(queryDesc))
    return;


  if (prev_ExecutorRun_hook)
    prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
  else
    standard_ExecutorRun(queryDesc, direction, count, execute_once);
}

void _PG_init(void)
{

  elog(NOTICE, "Installing custom executor hook...");
  prev_ExecutorRun_hook = ExecutorRun_hook;
  ExecutorRun_hook = custom_executor;
}

void _PG_fini(void)
{
  elog(NOTICE, "Uninstalling custom executor hook...");
  ExecutorRun_hook = NULL;
}
