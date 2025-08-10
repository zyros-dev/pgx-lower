#ifndef MLIR_DIALECT_DB_IR_DBOPS_H
#define MLIR_DIALECT_DB_IR_DBOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

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
using ::mlir::Value;
} // namespace pgx::mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.h.inc"
mlir::Type getBaseType(mlir::Type t);
bool isIntegerType(mlir::Type, unsigned int width);
int getIntegerWidth(mlir::Type, bool isUnSigned);
#endif // MLIR_DIALECT_DB_IR_DBOPS_H
