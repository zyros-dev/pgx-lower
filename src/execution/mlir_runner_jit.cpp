#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

// JIT execution infrastructure - heavy LLVM includes isolated here
#include "llvm/Support/TargetSelect.h"
#include "execution/jit_execution_interface.h"

// PostgreSQL execution includes
#ifndef BUILDING_UNIT_TESTS
extern "C" {
#include "postgres.h"
#include "executor/executor.h"
#include "nodes/execnodes.h"
#include "utils/errcodes.h"
#include "utils/elog.h"
}
#endif

// External JIT interface functions (from jit_execution_interface.h)
extern "C" {
    struct ModuleHandle* pgx_jit_create_module_handle(void* mlir_module_ptr);
    void pgx_jit_destroy_module_handle(struct ModuleHandle* handle);
    struct ExecutionHandle* pgx_jit_create_execution_handle(struct ModuleHandle* module_handle);
    void pgx_jit_destroy_execution_handle(struct ExecutionHandle* exec_handle);
    int pgx_jit_execute_query(struct ExecutionHandle* exec_handle, void* estate, void* dest);
    const char* pgx_jit_get_last_error(void);
    bool test_unit_code_from_postgresql();
}

namespace mlir_runner {

// JIT execution with PostgreSQL DestReceiver integration
bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, void* dest) {
    auto moduleHandle = pgx_jit_create_module_handle(&module);
    if (!moduleHandle) {
        PGX_ERROR("Failed to create module handle for JIT execution");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
    auto execHandle = pgx_jit_create_execution_handle(moduleHandle);
    pgx_jit_destroy_module_handle(moduleHandle);
    
    if (!execHandle) {
        PGX_ERROR("Failed to create JIT execution handle");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
    int result = pgx_jit_execute_query(execHandle, (void*)estate, dest);
    pgx_jit_destroy_execution_handle(execHandle);
    
    if (result != 0) {
        PGX_ERROR("JIT query execution failed with code: " + std::to_string(result));
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
    return true;
}

// Simplified MlirRunner class - now focused only on JIT execution
class MlirRunner {
public:
    bool executeQuery(::mlir::ModuleOp module, EState* estate, void* dest) {
        return executeJITWithDestReceiver(module, estate, dest);
    }
};

} // namespace mlir_runner