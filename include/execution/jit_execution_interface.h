#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations - avoid including any headers that might conflict
struct ModuleHandle;
struct ExecutionHandle;

ExecutionHandle* pgx_jit_create_execution_handle(ModuleHandle* module_handle);

int pgx_jit_execute_query(ExecutionHandle* exec_handle, void* estate, void* dest);

void pgx_jit_destroy_execution_handle(ExecutionHandle* exec_handle);

const char* pgx_jit_get_last_error(void);

#ifdef __cplusplus
}
#endif