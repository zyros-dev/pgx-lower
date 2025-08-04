#ifndef TUPLESTREAM_TYPES_H
#define TUPLESTREAM_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Forward declarations for TupleStream dialect types
namespace pgx_lower::compiler::dialect::tuples {
// TODO Phase 5: These should be generated from TableGen, but for now use forward declarations
class TupleStreamType;
class TupleType;

// TODO Phase 5: Generate from TupleStreamBase.td when TableGen setup is complete
// For now, provide minimal interface to get compilation working
}

#endif // TUPLESTREAM_TYPES_H