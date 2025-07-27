#ifndef TUPLESTREAM_DIALECT_H
#define TUPLESTREAM_DIALECT_H

#include "mlir/IR/Dialect.h"

// Forward declare ColumnManager for now - TODO: Port from LingoDB if needed
class ColumnManager;

#include "TupleStreamDialect.h.inc"

// Include types from TableGen
#define GET_TYPEDEF_CLASSES  
#include "TupleStreamTypes.h.inc"

#endif // TUPLESTREAM_DIALECT_H