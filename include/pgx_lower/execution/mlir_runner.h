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

// For DestReceiver, we need conditional compilation
#ifndef POSTGRESQL_EXTENSION
// In unit tests, use void* for DestReceiver
typedef void* DestReceiver;
#endif

// Forward declarations for MLIR
namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

// MLIR Context Management
bool initialize_mlir_context(::mlir::MLIRContext& context);

// JIT Execution (forward declaration only - avoid including complex MLIR types)
bool executeJITWithDestReceiver(mlir::ModuleOp module, EState* estate, DestReceiver* dest);

// PostgreSQL Integration - New AST-based translation (replaces ColumnExpression approach)
auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt) -> bool;

// PostgreSQL Integration with EState memory context support
auto run_mlir_with_estate(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext) -> bool;

// PostgreSQL Integration with DestReceiver for result streaming (Phase 4g-2c)
// Note: DestReceiver forward declaration is handled in the implementation file to avoid header conflicts

} // namespace mlir_runner

#endif // MLIR_RUNNER_H