#ifndef PGX_DIALECT_DB_IR_DBOPS_H
#define PGX_DIALECT_DB_IR_DBOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.h.inc"

#endif // PGX_DIALECT_DB_IR_DBOPS_H