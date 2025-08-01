//===- DSADialect.h - Data Structure Access dialect --------------*- C++ -*-===//
//
// Data Structure Access dialect for low-level data access patterns
//
//===----------------------------------------------------------------------===//

#ifndef DSA_DIALECT_H
#define DSA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "DSADialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "DSATypes.h.inc"

#define GET_OP_CLASSES
#include "DSAOps.h.inc"

#endif // DSA_DIALECT_H