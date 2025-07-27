#ifndef TUPLESTREAM_DIALECT_H
#define TUPLESTREAM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "Column.h"

// Forward declare ColumnManager for now - TODO: Port from LingoDB if needed
class ColumnManager;

#include "TupleStreamDialect.h.inc"

// Include types and attributes from TableGen
#define GET_TYPEDEF_CLASSES  
#include "TupleStreamTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.h.inc"

#endif // TUPLESTREAM_DIALECT_H