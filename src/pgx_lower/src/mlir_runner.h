#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include "mlir_logger.h"
#include <cstdint>
#include <functional>

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

// PostgreSQL Integration - Main entry point called by my_executor.cpp
bool run_mlir_postgres_table_scan(const char* tableName, MLIRLogger& logger);

// Core MLIR compilation and execution engine
bool run_mlir_core(int64_t intValue, MLIRLogger& logger);

// Unit Test Interface - Only available when building as standalone library
#ifndef POSTGRESQL_EXTENSION
bool run_mlir_test(int64_t intValue);
#endif

}  // namespace mlir_runner

#endif  // MLIR_RUNNER_H