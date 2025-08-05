#ifndef PGX_LOWER_MLIR_DIALECT_RELALG_IR_RELALGOPS_H
#define PGX_LOWER_MLIR_DIALECT_RELALG_IR_RELALGOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"  
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"

// Forward declarations for generated code - avoid using declarations that interfere with TableGen

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h.inc"

#define GET_OP_CLASSES  
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h.inc"

#endif // PGX_LOWER_MLIR_DIALECT_RELALG_IR_RELALGOPS_H