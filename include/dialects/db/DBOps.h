#ifndef DB_OPS_H
#define DB_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// #include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h" // Removed Arrow dependency
// #include "DBOpsEnums.h.inc" // TODO: Generate if needed
// #include "lingodb/compiler/Dialect/DB/IR/DBOpsInterfaces.h" // TODO: Port if needed
#include "DBTypes.h"
// #include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h" // TODO: Port if needed

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "DBOps.h.inc"
// TODO: Port these utility functions if needed
// mlir::Type getBaseType(mlir::Type t);
// mlir::Type wrapNullableType(mlir::MLIRContext* context, mlir::Type type, mlir::ValueRange values);
// bool isIntegerType(mlir::Type, unsigned int width);
// int getIntegerWidth(mlir::Type, bool isUnSigned);
#endif //DB_OPS_H
