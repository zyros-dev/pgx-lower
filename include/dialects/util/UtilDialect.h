#ifndef UTIL_DIALECT_H
#define UTIL_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "UtilDialect.h.inc"

// Include types from TableGen
#define GET_TYPEDEF_CLASSES
#include "UtilTypes.h.inc"

#endif // UTIL_DIALECT_H
