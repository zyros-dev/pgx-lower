#ifndef PGX_DIALECT_UTIL_IR_UTILOPS_H
#define PGX_DIALECT_UTIL_IR_UTILOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Util/IR/UtilOps.h.inc"

#endif // PGX_DIALECT_UTIL_IR_UTILOPS_H