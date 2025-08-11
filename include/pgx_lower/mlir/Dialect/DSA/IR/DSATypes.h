#ifndef MLIR_DIALECT_DSA_IR_DSATYPES_H
#define MLIR_DIALECT_DSA_IR_DSATYPES_H

#include "pgx_lower/mlir/Dialect/DSA/IR/DSACollectionType.h"

#include "pgx_lower/mlir/Dialect/DSA/IR/DSAOpsEnums.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.h.inc"

#endif // MLIR_DIALECT_DSA_IR_DSATYPES_H
