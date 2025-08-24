#pragma once

#include <unordered_map>
#include <string>
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"

// Forward declarations to avoid PostgreSQL header inclusion
struct PlannedStmt;
namespace mlir {
    class OpBuilder;
}

namespace pgx_lower::frontend::sql {

// Column information structure for schema discovery
struct ColumnInfo {
    std::string name;
    unsigned int typeOid;  // PostgreSQL type OID
    int32_t typmod;        // Type modifier
    bool nullable;

    ColumnInfo(std::string n, unsigned int t, int32_t m, bool null)
        : name(std::move(n)), typeOid(t), typmod(m), nullable(null) {}
};

// Translation context for managing state during AST translation
struct TranslationContext {
    PlannedStmt* currentStmt = nullptr;
    ::mlir::OpBuilder* builder = nullptr;
    std::unordered_map<unsigned int, ::mlir::Type> typeCache;  // Oid -> Type mapping
    ::mlir::Value currentTuple = nullptr;
    // TODO: Add column mapping when BaseTableOp attribute printing is fixed
};

} // namespace pgx_lower::frontend::sql