#ifndef TUPLESTREAM_DIALECT_H
#define TUPLESTREAM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "Column.h"
#include "ColumnManager.h"

namespace pgx_lower::compiler::dialect::tuples {
class TupleStreamDialect;
} // namespace pgx_lower::compiler::dialect::tuples

#include "TupleStreamDialect.h.inc"

// Include types and attributes from TableGen
#define GET_TYPEDEF_CLASSES  
#include "TupleStreamTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.h.inc"


#endif // TUPLESTREAM_DIALECT_H