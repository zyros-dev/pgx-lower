//===- SubOpDialect.h - SubOperator dialect ----------------------*- C++ -*-===//
//
// SubOperator dialect for tuple stream operations
//
//===----------------------------------------------------------------------===//

#ifndef SUBOP_DIALECT_H
#define SUBOP_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SubOpDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "SubOpTypes.h.inc"

#define GET_OP_CLASSES
#include "SubOpOps.h.inc"

#endif // SUBOP_DIALECT_H