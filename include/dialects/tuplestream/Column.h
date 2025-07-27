#ifndef TUPLESTREAM_COLUMN_H
#define TUPLESTREAM_COLUMN_H

#include <string>
#include <memory>
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

namespace pgx_lower::compiler::dialect::tuples {
// PGX_LOWER IMPLEMENTATION: This is adapted from LingoDB's column-oriented design
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

} // namespace pgx_lower::compiler::dialect::tuples

// CRITICAL: Declare hash_value IMMEDIATELY after Column definition
// to ensure it's available during template instantiation
namespace llvm {
inline llvm::hash_code hash_value(const std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>& arg) {
    return llvm::hash_value(arg.get());
}
} // namespace llvm

// Hash support for shared_ptr<Column> needed by MLIR
namespace std {
template <>
struct hash<std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>> {
    size_t operator()(const std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>& ptr) const {
        return std::hash<void*>{}(ptr.get());
    }
};
} // namespace std

namespace llvm {
template <>
struct DenseMapInfo<std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>> {
    static std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column> getEmptyKey() {
        return std::make_shared<pgx_lower::compiler::dialect::tuples::Column>();
    }
    static std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column> getTombstoneKey() {
        static auto tombstone = std::make_shared<pgx_lower::compiler::dialect::tuples::Column>();
        return tombstone;
    }
    static unsigned getHashValue(const std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>& val) {
        if (!val) return 0;
        return static_cast<unsigned>(hash_value(val));
    }
    static bool isEqual(const std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>& lhs,
                       const std::shared_ptr<pgx_lower::compiler::dialect::tuples::Column>& rhs) {
        return lhs == rhs; // Just compare pointers for LingoDB approach
    }
};
} // namespace llvm

#endif //TUPLESTREAM_COLUMN_H