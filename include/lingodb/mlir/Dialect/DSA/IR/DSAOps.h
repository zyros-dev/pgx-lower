#ifndef MLIR_DIALECT_DSA_IR_DSAOPS_H
#define MLIR_DIALECT_DSA_IR_DSAOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lingodb/mlir/Dialect/DSA/IR/DSACollectionType.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOpsEnums.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOpsInterfaces.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h.inc"
mlir::Type getBaseType(mlir::Type t);
bool isIntegerType(mlir::Type, unsigned int width);
int getIntegerWidth(mlir::Type, bool isUnSigned);
#endif // MLIR_DIALECT_DSA_IR_DSAOPS_H
