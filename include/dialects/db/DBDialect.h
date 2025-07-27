//===- DBDialect.h - Database dialect ----------------------------*- C++ -*-===//
//
// Database dialect for type-polymorphic operations with NULL handling
//
//===----------------------------------------------------------------------===//

#ifndef DB_DIALECT_H
#define DB_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "DBDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "DBTypes.h.inc"

#define GET_OP_CLASSES
#include "DBOps.h.inc"

#endif // DB_DIALECT_H