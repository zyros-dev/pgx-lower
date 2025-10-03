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
#else
// Forward declaration for PostgreSQL extension - use the same structure as PostgreSQL
struct _DestReceiver;
typedef struct _DestReceiver DestReceiver;
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
bool setupMLIRContextForJIT(::mlir::MLIRContext& context);

bool executeJITWithDestReceiver(mlir::ModuleOp module, EState* estate, DestReceiver* dest);

auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt) -> bool;

auto run_mlir_with_estate(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext) -> bool;

// MLIR Pipeline Phases - for testing access
bool runPhase3a(::mlir::ModuleOp module);
bool runPhase3b(::mlir::ModuleOp module);
bool runPhase3c(::mlir::ModuleOp module);
bool runCompleteLoweringPipeline(::mlir::ModuleOp module);

} // namespace mlir_runner

#endif // MLIR_RUNNER_H