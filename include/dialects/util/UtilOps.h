#ifndef UTIL_OPS_H
#define UTIL_OPS_H

#include "dialects/util/UtilTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "UtilOps.h.inc"

#endif //UTIL_OPS_H
