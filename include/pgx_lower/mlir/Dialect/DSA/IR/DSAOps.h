#ifndef PGX_DIALECT_DSA_IR_DSAOPS_H
#define PGX_DIALECT_DSA_IR_DSAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include <optional>

#include "mlir/Dialect/DSA/IR/DSADialect.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOps.h.inc"

#endif // PGX_DIALECT_DSA_IR_DSAOPS_H