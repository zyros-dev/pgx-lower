#pragma once

#include <memory>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Target/TargetMachine.h"

// Process-lifetime MLIR/LLVM runtime state. Constructed on first use,
// destroyed via shutdown_mlir_runtime() at _PG_fini.
//
// Thread safety: the PostgreSQL backend is single-threaded per process;
// parallel workers don't go through the executor hook. Do not call from a
// parallel worker.
//
// Ownership: target_machine is owned by the singleton and outlives every
// JITEngine. mlir::ExecutionEngine does not take ownership of an
// externally-supplied TargetMachine, so passing target_machine.get() into
// the optimizer pipeline is safe.
//
// Per-query state that is NOT hoisted here (still reset per query by the
// executor / runtime layer):
//   - g_execution_context (runtime/PostgreSQLRuntime.cpp)
//   - g_tuple_streamer (runtime/tuple_access.cpp)
//   - g_current_tuple_passthrough, g_computed_results, g_jit_table_oid
//     (runtime/tuple_access.cpp)
//   - PassManager instances (mlir_setup/mlir_runner_phases.cpp) — cheap
//     relative to LLVM codegen; spec 03 will bypass them via caching.

namespace pgx_lower::execution {

struct MLIRRuntime {
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    std::unique_ptr<llvm::TargetMachine> target_machine;
};

MLIRRuntime& get_mlir_runtime();
void shutdown_mlir_runtime();

}  // namespace pgx_lower::execution

extern "C" void initialize_mlir_runtime(void);
extern "C" void shutdown_mlir_runtime_c(void);
