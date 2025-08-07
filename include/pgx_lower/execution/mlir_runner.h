#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include <cstdint>
#include <functional>
#include <vector>

// Forward declarations

// PostgreSQL C headers - need extern "C" wrapping
extern "C" {
struct PlannedStmt;
struct EState;
struct ExprContext;
}

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

// PostgreSQL Integration - New AST-based translation (replaces ColumnExpression approach)
auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt) -> bool;

// PostgreSQL Integration with EState memory context support
auto run_mlir_with_estate(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext) -> bool;

} // namespace mlir_runner

#endif // MLIR_RUNNER_H