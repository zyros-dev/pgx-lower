#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include "mlir_logger.h"
#include <cstdint>
#include <functional>
#include <vector>

// Forward declarations

// PostgreSQL C headers - need extern "C" wrapping
extern "C" {
struct PlannedStmt;
}

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

// PostgreSQL Integration - New AST-based translation (replaces ColumnExpression approach)
auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) -> bool;

} // namespace mlir_runner

#endif // MLIR_RUNNER_H