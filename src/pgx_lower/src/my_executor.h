#ifndef EXECUTOR_H
#define EXECUTOR_H

// Prevent libintl.h conflicts with PostgreSQL macros
#define ENABLE_NLS 0

// Include MLIR headers first to avoid conflicts
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h.inc>

// Example PostgreSQL executor header file

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "nodes/execnodes.h"
#include <executor/execdesc.h>
#include "access/relscan.h"
#include "catalog/pg_type.h"

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

struct MyCppPlan {};

auto run_mlir(int64_t intValue) -> void;
bool run_mlir_with_tuple_scan(TableScanDesc scanDesc, TupleDesc tupdesc, const QueryDesc* queryDesc);

class MyCppExecutor {
   public:
    bool execute(const QueryDesc* plan);
};

#endif // __cplusplus

#endif // EXECUTOR_H
