#ifndef DB_TYPES_H
#define DB_TYPES_H

// #include "lingodb/compiler/Dialect/DB/IR/DBOpsEnums.h" // TODO: Port if needed
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "DBTypes.h.inc"

#endif //DB_TYPES_H
