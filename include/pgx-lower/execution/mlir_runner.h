#ifndef MLIR_RUNNER_H
#define MLIR_RUNNER_H

#include <cstdint>
#include <functional>
#include <vector>

// Forward declarations
extern "C" {
struct PlannedStmt;
struct EState;
struct ExprContext;
}

#ifndef POSTGRESQL_EXTENSION
typedef void* DestReceiver;
#else
struct _DestReceiver;
typedef struct _DestReceiver DestReceiver;
#endif

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace mlir_runner {

using ExternalFunction = std::function<int64_t()>;

bool initialize_mlir_context(::mlir::MLIRContext& context);
bool setupMLIRContextForJIT(::mlir::MLIRContext& context);
bool executeJITWithDestReceiver(mlir::ModuleOp module, EState* estate, DestReceiver* dest);
auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt) -> bool;
auto run_mlir_with_estate(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext) -> bool;
bool runPhase3a(::mlir::ModuleOp module);
bool runPhase3b(::mlir::ModuleOp module);
bool runPhase3c(::mlir::ModuleOp module);
bool runCompleteLoweringPipeline(::mlir::ModuleOp module);

bool verifyModuleOrThrow(::mlir::ModuleOp module, const char* phase_name, const char* error_context);

} // namespace mlir_runner

#endif // MLIR_RUNNER_H