#ifndef PGX_DIALECT_DSA_IR_DSATYPES_H
#define PGX_DIALECT_DSA_IR_DSATYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

// Forward declare TupleType properly
using TupleType = ::mlir::TupleType;

#define GET_TYPEDEF_CLASSES  
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.h.inc"

#endif // PGX_DIALECT_DSA_IR_DSATYPES_H