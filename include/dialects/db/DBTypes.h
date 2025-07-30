#ifndef DB_TYPES_H
#define DB_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Include enum definitions before type definitions
#include "DBOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "DBTypes.h.inc"

#endif //DB_TYPES_H
