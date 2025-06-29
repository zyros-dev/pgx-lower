#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include "mlir_logger.h"
#include <cstdint>
#include <functional>
#include <vector>

// Forward declaration to avoid circular dependency
namespace mlir_builder {
    struct ColumnExpression;
}

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

// PostgreSQL Integration - Main entry point called by my_executor.cpp
auto run_mlir_postgres_table_scan(const char* tableName, MLIRLogger& logger) -> bool;

// PostgreSQL Integration - Typed field access with pg dialect
auto run_mlir_postgres_typed_table_scan(const char* tableName, MLIRLogger& logger) -> bool;

// PostgreSQL Integration - Typed field access with specific columns
auto run_mlir_postgres_typed_table_scan_with_columns(const char* tableName,
                                                     const std::vector<mlir_builder::ColumnExpression>& expressions,
                                                     MLIRLogger& logger) -> bool;

// PostgreSQL Integration - Typed field access with WHERE clause support
auto run_mlir_postgres_typed_table_scan_with_where(const char* tableName,
                                                   const std::vector<mlir_builder::ColumnExpression>& expressions,
                                                   const mlir_builder::ColumnExpression& whereClause,
                                                   MLIRLogger& logger) -> bool;



// Core MLIR compilation and execution engine
auto run_mlir_core(int64_t intValue, MLIRLogger& logger) -> bool;

// Unit Test Interface - Only available when building as standalone library
#ifndef POSTGRESQL_EXTENSION
auto run_mlir_test(int64_t intValue) -> bool;
#endif

} // namespace mlir_runner

#endif // MLIR_RUNNER_H