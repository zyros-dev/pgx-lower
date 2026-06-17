#ifndef MLIR_DIALECT_DB_IR_DBOPSATTRIBUTES_H
#define MLIR_DIALECT_DB_IR_DBOPSATTRIBUTES_H

#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "lingodb/mlir/Dialect/DB/IR/DBPgTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/ArrayRef.h"

#define GET_ATTRDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsAttributes.h.inc"

#endif // MLIR_DIALECT_DB_IR_DBOPSATTRIBUTES_H
