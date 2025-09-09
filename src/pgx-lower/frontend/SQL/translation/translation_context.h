#pragma once

#include <unordered_map>
#include <map>
#include <string>
#include <memory>
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"

// Forward declarations to avoid PostgreSQL header inclusion
struct PlannedStmt;
namespace mlir {
    class OpBuilder;
}

namespace pgx_lower::frontend::sql {

// Type aliases for column mapping clarity
using varno_t = int;
using varattno_t = int;
using table_t = std::string;
using column_t = std::string;
using ColumnMapping = std::map<std::pair<varno_t, varattno_t>, std::pair<table_t, column_t>>;

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
    ColumnMapping columnMappings;  // Maps (varno, varattno) -> (table_name, column_name)
};

} // namespace pgx_lower::frontend::sql