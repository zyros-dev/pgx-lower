#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations - avoid including any headers that might conflict
struct ModuleHandle;
struct ExecutionHandle;

/**
 * Create a JIT execution handle for a compiled MLIR module
 * This function is implemented in jit_execution_wrapper.cpp which isolates LLVM headers
 * 
 * @param module_handle Opaque handle to MLIR module
 * @return Opaque handle to execution engine, or NULL on failure
 */
ExecutionHandle* pgx_jit_create_execution_handle(ModuleHandle* module_handle);

/**
 * Execute a JIT compiled query
 * 
 * @param exec_handle Opaque handle to execution engine
 * @param estate PostgreSQL execution state (void* to avoid header inclusion)
 * @param dest Destination receiver (void* to avoid header inclusion)
 * @return 0 on success, non-zero on failure
 */
int pgx_jit_execute_query(ExecutionHandle* exec_handle, void* estate, void* dest);

/**
 * Cleanup JIT execution handle
 * 
 * @param exec_handle Handle to destroy
 */
void pgx_jit_destroy_execution_handle(ExecutionHandle* exec_handle);

/**
 * Get last error message from JIT execution
 * 
 * @return Error message string, or NULL if no error
 */
const char* pgx_jit_get_last_error(void);

#ifdef __cplusplus
}
#endif