#ifndef PGX_LOWER_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
#define PGX_LOWER_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H

#include "dialects/tuplestream/Column.h"
#include "TupleStreamAttrs.h.inc"
#include "TupleStreamTypes.h.inc"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "TupleStreamOps.h.inc"

#endif //PGX_LOWER_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
