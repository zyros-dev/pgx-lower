#ifndef MLIR_DIALECT_RELALG_IR_RELALGOPS_H
#define MLIR_DIALECT_RELALG_IR_RELALGOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"

#include "mlir/Dialect/RelAlg/IR/Column.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

// LLVM 20 compatibility: The generated code uses unqualified mlir:: types
// within the pgx::mlir namespace, which causes lookup issues
namespace pgx::mlir {
using ::mlir::Attribute;
using ::mlir::MLIRContext;
using ::mlir::IntegerAttr;
using ::mlir::StringAttr;
using ::mlir::BoolAttr;
using ::mlir::ArrayAttr;
using ::mlir::DictionaryAttr;
} // namespace pgx::mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h.inc"

#endif // MLIR_DIALECT_RELALG_IR_RELALGOPS_H
