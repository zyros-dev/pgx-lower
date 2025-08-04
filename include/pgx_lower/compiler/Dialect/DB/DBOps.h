#ifndef DB_OPS_H
#define DB_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// #include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h" // Removed Arrow dependency
#include "DBTypes.h" // DBTypes.h already includes DBOpsEnums.h.inc
#include "DBOpsInterfaces.h.inc"
#include "compiler/Dialect/RelAlg/RelAlgInterfaces.h" // For CmpOpInterface
#include "compiler/Dialect/util/UtilTypes.h" // For RefType

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "DBOps.h.inc"
// Utility functions for DB operations
mlir::Type getBaseType(mlir::Type t);
mlir::Type wrapNullableType(mlir::MLIRContext* context, mlir::Type type, mlir::ValueRange values);
bool isIntegerType(mlir::Type, unsigned int width);
int getIntegerWidth(mlir::Type, bool isUnSigned);

// Type inference functions needed by TableGen
mlir::LogicalResult inferReturnType(mlir::MLIRContext* context, std::optional<mlir::Location> location,
                                   mlir::ValueRange operands, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
mlir::LogicalResult inferArithmeticReturnType(mlir::MLIRContext* context, std::optional<mlir::Location> location,
                                            mlir::ValueRange operands, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
mlir::LogicalResult inferMulReturnType(mlir::MLIRContext* context, std::optional<mlir::Location> location,
                                      mlir::ValueRange operands, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
mlir::LogicalResult inferDivReturnType(mlir::MLIRContext* context, std::optional<mlir::Location> location,
                                      mlir::ValueRange operands, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
mlir::LogicalResult inferRemReturnType(mlir::MLIRContext* context, std::optional<mlir::Location> location,
                                      mlir::ValueRange operands, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
#endif //DB_OPS_H
