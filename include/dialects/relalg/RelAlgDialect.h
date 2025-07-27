#ifndef RELALG_DIALECT_H
#define RELALG_DIALECT_H

#include "mlir/IR/Dialect.h"

#include "RelAlgDialect.h.inc"

// Include types from TableGen
#define GET_TYPEDEF_CLASSES
#include "RelAlgTypes.h.inc"

#endif // RELALG_DIALECT_H
