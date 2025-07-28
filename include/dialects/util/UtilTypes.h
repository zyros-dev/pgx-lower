#ifndef UTIL_TYPES_H
#define UTIL_TYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "dialects/util/UtilDialect.h"

// Include type interface declarations
#include "UtilOpsTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "UtilTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

#endif //UTIL_TYPES_H
