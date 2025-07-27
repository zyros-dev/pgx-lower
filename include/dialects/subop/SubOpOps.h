#ifndef SUBOP_OPS_H
#define SUBOP_OPS_H

// #include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h" // Removed Arrow dependency
// #include "dialects/db/DBTypes.h" // TODO: Fix circular dependency
#include "SubOpInterfaces.h"
// #include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h" // TODO: Port if needed
// #include "SubOpOpsEnums.h.inc" // TODO: Generate if needed
#include "SubOpTypes.h.inc"
#include "dialects/tuplestream/Column.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
// #include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h" // TODO: Port if needed

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "SubOpOps.h.inc"

#endif //SUBOP_OPS_H
