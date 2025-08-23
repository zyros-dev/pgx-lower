#ifndef MLIR_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H
#define MLIR_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "lingodb/runtime/metadata.h"

#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#define GET_ATTRDEF_CLASSES
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h.inc"

#endif // MLIR_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H