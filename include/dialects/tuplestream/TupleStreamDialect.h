#ifndef TUPLESTREAM_DIALECT_H
#define TUPLESTREAM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "Column.h"
#include "ColumnManager.h"

namespace pgx_lower::compiler::dialect::tuples {
class TupleStreamDialect;
} // namespace pgx_lower::compiler::dialect::tuples

#include "TupleStreamDialect.h.inc"

// Include types from TableGen with guard
#ifndef TUPLESTREAM_TYPES_INCLUDED
#define TUPLESTREAM_TYPES_INCLUDED
#define GET_TYPEDEF_CLASSES  
#include "TupleStreamTypes.h.inc"
#endif

// Attributes are already included via ColumnManager.h


#endif // TUPLESTREAM_DIALECT_H