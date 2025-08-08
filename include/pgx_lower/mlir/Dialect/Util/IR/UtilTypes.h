#ifndef PGX_DIALECT_UTIL_IR_UTILTYPES_H
#define PGX_DIALECT_UTIL_IR_UTILTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/StorageUniquerSupport.h"

#define GET_TYPEDEF_CLASSES  
#include "mlir/Dialect/Util/IR/UtilOpsTypes.h.inc"

#endif // PGX_DIALECT_UTIL_IR_UTILTYPES_H