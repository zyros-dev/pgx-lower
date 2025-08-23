#ifndef EXECUTOR_H
#define EXECUTOR_H

#define ENABLE_NLS 0

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h.inc>

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

struct TupleScanContext;
struct PostgreSQLTuplePassthrough;
struct TupleStreamer;

class MyCppExecutor {
   public:
    static bool execute(const QueryDesc* plan);
};

#endif // __cplusplus

#endif // EXECUTOR_H
