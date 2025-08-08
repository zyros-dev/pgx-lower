#ifndef PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMN_H
#define PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMN_H

#include <mlir/IR/Types.h>
#include <string>

namespace pgx {
namespace mlir {
namespace relalg {

// Column represents a single column in a relation with its type information
// This matches the LingoDB pattern where Column contains just the type
struct Column {
    ::mlir::Type type;
    
    Column() = default;
    explicit Column(::mlir::Type t) : type(t) {}
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_DIALECT_RELALG_IR_COLUMN_H