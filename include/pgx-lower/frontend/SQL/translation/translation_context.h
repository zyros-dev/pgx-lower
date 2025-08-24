#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// MLIR includes needed for Value type
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Operation.h"

// Forward declarations to avoid PostgreSQL header includes
extern "C" {
struct List;
struct RangeTblEntry;
typedef unsigned int Oid;
typedef uintptr_t Datum;
}

namespace mlir {
class MLIRContext;
class OpBuilder;
class Location;
class Block;
namespace func {
class FuncOp;
}
}

namespace pgx_lower::ast {

// Shared context for all translation operations
struct TranslationContext {
    ::mlir::MLIRContext* mlir_context;
    ::mlir::OpBuilder* builder;
    ::mlir::func::FuncOp* current_function;
    ::mlir::Block* current_block;
    
    // PostgreSQL query context
    List* rtable;  // Range table from query
    List* targetList;  // Target list from query
    
    // Type system cache
    std::unordered_map<Oid, ::mlir::Type> type_cache;
    
    // Schema information
    struct ColumnInfo {
        std::string name;
        Oid type_oid;
        int32_t typmod;
        bool nullable;
    };
    std::vector<ColumnInfo> scan_columns;
    
    // Operation tracking
    ::mlir::Operation* last_operation = nullptr;
    bool in_aggregate_context = false;
};

// Result types for translation functions
struct ExpressionResult {
    ::mlir::Value value;
    bool is_nullable = false;
};

struct PlanResult {
    ::mlir::Operation* operation;
    std::vector<::mlir::Value> output_values;
};

} // namespace pgx_lower::ast