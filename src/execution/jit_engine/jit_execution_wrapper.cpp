// It isolates LLVM headers from PostgreSQL headers to avoid conflicts

#include "execution/jit_execution_interface.h"
#include "execution/jit_execution_engine.h"
#include "execution/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <string>

// Thread-local error message storage
static thread_local std::string last_error_message;

// Opaque handle structures
struct ModuleHandle {
    ::mlir::ModuleOp module;
    ModuleHandle(::mlir::ModuleOp m) : module(m) {}
};

struct ExecutionHandle {
    std::unique_ptr<pgx_lower::execution::PostgreSQLJITExecutionEngine> engine;
    ::mlir::ModuleOp module;
    
    ExecutionHandle() : engine(std::make_unique<pgx_lower::execution::PostgreSQLJITExecutionEngine>()) {}
};

extern "C" {

ExecutionHandle* pgx_jit_create_execution_handle(ModuleHandle* module_handle) {
    if (!module_handle) {
        last_error_message = "Null module handle provided";
        PGX_ERROR("pgx_jit_create_execution_handle: " + last_error_message);
        return nullptr;
    }
    
    try {
        auto exec_handle = new ExecutionHandle();
        exec_handle->module = module_handle->module;
        
        if (!exec_handle->engine->initialize(module_handle->module)) {
            last_error_message = "Failed to initialize JIT execution engine";
            PGX_ERROR("pgx_jit_create_execution_handle: " + last_error_message);
            delete exec_handle;
            return nullptr;
        }
        
        exec_handle->engine->registerPostgreSQLRuntimeFunctions();
        
        if (!exec_handle->engine->setupMemoryContexts()) {
            last_error_message = "Failed to setup memory contexts";
            PGX_ERROR("pgx_jit_create_execution_handle: " + last_error_message);
            delete exec_handle;
            return nullptr;
        }
        
        PGX_INFO("JIT execution handle created successfully");
        return exec_handle;
        
    } catch (const std::exception& e) {
        last_error_message = std::string("Exception creating JIT handle: ") + e.what();
        PGX_ERROR("pgx_jit_create_execution_handle: " + last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception creating JIT handle";
        PGX_ERROR("pgx_jit_create_execution_handle: " + last_error_message);
        return nullptr;
    }
}

int pgx_jit_execute_query(ExecutionHandle* exec_handle, void* estate, void* dest) {
    if (!exec_handle) {
        last_error_message = "Null execution handle";
        PGX_ERROR("pgx_jit_execute_query: " + last_error_message);
        return -1;
    }
    
    if (!estate || !dest) {
        last_error_message = "Null estate or destination receiver";
        PGX_ERROR("pgx_jit_execute_query: " + last_error_message);
        return -2;
    }
    
    try {
        bool success = exec_handle->engine->executeCompiledQuery(estate, dest);
        
        if (!success) {
            last_error_message = "JIT query execution failed";
            PGX_ERROR("pgx_jit_execute_query: " + last_error_message);
            return -3;
        }
        
        PGX_INFO("JIT query execution completed successfully");
        return 0;
        
    } catch (const std::exception& e) {
        last_error_message = std::string("Exception during JIT execution: ") + e.what();
        PGX_ERROR("pgx_jit_execute_query: " + last_error_message);
        return -4;
    } catch (...) {
        last_error_message = "Unknown exception during JIT execution";
        PGX_ERROR("pgx_jit_execute_query: " + last_error_message);
        return -5;
    }
}

void pgx_jit_destroy_execution_handle(ExecutionHandle* exec_handle) {
    if (exec_handle) {
        PGX_DEBUG("Destroying JIT execution handle");
        delete exec_handle;
    }
}

const char* pgx_jit_get_last_error(void) {
    return last_error_message.empty() ? nullptr : last_error_message.c_str();
}

} // extern "C"

extern "C" ModuleHandle* pgx_jit_create_module_handle(void* mlir_module_ptr) {
    if (!mlir_module_ptr) {
        last_error_message = "Null MLIR module pointer";
        return nullptr;
    }
    
    try {
        auto module = static_cast<::mlir::ModuleOp*>(mlir_module_ptr);
        return new ModuleHandle(*module);
    } catch (...) {
        last_error_message = "Failed to create module handle";
        return nullptr;
    }
}

extern "C" void pgx_jit_destroy_module_handle(ModuleHandle* handle) {
    delete handle;
}