#ifndef PGX_LOWER_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
#define PGX_LOWER_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H

#include "dialects/tuplestream/Column.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"


// Include generated definitions outside namespace with guard
#ifndef TUPLESTREAM_TYPES_INCLUDED
#define TUPLESTREAM_TYPES_INCLUDED
#define GET_TYPEDEF_CLASSES
#include "TupleStreamTypes.h.inc"
#endif

// Include attributes with guard to prevent duplicates
#ifndef TUPLESTREAM_ATTRS_INCLUDED
#define TUPLESTREAM_ATTRS_INCLUDED
#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.h.inc"
#endif

#define GET_OP_CLASSES
#include "TupleStreamOps.h.inc"

#endif //PGX_LOWER_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
