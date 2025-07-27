#ifndef TUPLESTREAM_COLUMN_H
#define TUPLESTREAM_COLUMN_H

#include "mlir/IR/Types.h"
#include <string>

namespace mlir::tuples {
// TODO: This is adapted from LingoDB's column-oriented design
// Need to rethink for PostgreSQL's tuple-oriented approach

/// Column represents metadata for a column in a tuple stream
/// This is a minimal implementation for compilation compatibility
struct Column {
    mlir::Type type;
    std::string name;
    
    Column() = default;
    Column(mlir::Type t, const std::string& n = "") : type(t), name(n) {}
    
    // TODO: Add PostgreSQL-specific tuple field metadata
    // TODO: Add proper column management for PostgreSQL's tuple-oriented approach
};

} // namespace mlir::tuples

#endif //TUPLESTREAM_COLUMN_H