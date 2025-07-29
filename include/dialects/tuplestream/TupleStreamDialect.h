#ifndef TUPLESTREAM_DIALECT_H
#define TUPLESTREAM_DIALECT_H
#include <memory>

namespace llvm {
class hash_code; // NOLINT (readability-identifier-naming)
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg); // NOLINT (readability-identifier-naming)
} // end namespace llvm
#include "ColumnManager.h"
#include "mlir/IR/Dialect.h"
#include "Column.h"

#ifndef MLIR_HASHCODE_SHARED_PTR
#define MLIR_HASHCODE_SHARED_PTR
namespace llvm {
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg) { // NOLINT (readability-identifier-naming)
   return hash_value(arg.get());
}
} // end namespace llvm
#endif // MLIR_HASHCODE_SHARED_PTR

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