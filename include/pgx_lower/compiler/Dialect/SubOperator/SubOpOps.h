#ifndef SUBOP_OPS_H
#define SUBOP_OPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

// #include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h" // Removed Arrow dependency
// #include "compiler/Dialect/DB/DBTypes.h" // TODO Phase 5: Fix circular dependency

// Include enums first
#include "SubOpOpsEnums.h.inc"

// Include attributes that may use enums
#define GET_ATTRDEF_CLASSES
#include "SubOpOpsAttributes.h.inc"

// Include interfaces after attributes
#include "SubOpInterfaces.h"

// Include type interfaces
#include "SubOpTypeInterfaces.h.inc"

// Include types
#define GET_TYPEDEF_CLASSES
#include "SubOpTypes.h.inc"

#include "compiler/Dialect/TupleStream/Column.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
// #include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h" // TODO Phase 5: Port if needed

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "SubOpOps.h.inc"

#endif //SUBOP_OPS_H
