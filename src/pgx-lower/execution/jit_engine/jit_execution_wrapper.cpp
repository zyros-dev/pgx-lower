// It isolates LLVM headers from PostgreSQL headers to avoid conflicts

#include "pgx-lower/execution/jit_execution_interface.h"
#include "pgx-lower/execution/jit_execution_engine.h"
#include "pgx-lower/utility/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CodeGen.h"
#include <memory>
#include <string>

// Thread-local error message storage
static thread_local std::string last_error_message;

// Opaque handle structures
struct ModuleHandle {
    ::mlir::ModuleOp module;
    ModuleHandle(::mlir::ModuleOp m)
    : module(m) {}
};

struct ExecutionHandle {
    std::unique_ptr<pgx_lower::execution::PostgreSQLJITExecutionEngine> engine;
    ::mlir::ModuleOp module;

    ExecutionHandle()
    : engine(std::make_unique<pgx_lower::execution::PostgreSQLJITExecutionEngine>()) {}
};

extern "C" {

ExecutionHandle* pgx_jit_create_execution_handle(ModuleHandle* module_handle) {
    if (!module_handle) {
        last_error_message = "Null module handle provided";
        PGX_ERROR("pgx_jit_create_execution_handle: %s", last_error_message.c_str());
        return nullptr;
    }

    try {
        auto exec_handle = new ExecutionHandle();
        exec_handle->module = module_handle->module;

        exec_handle->engine->setOptimizationLevel(llvm::CodeGenOptLevel::Default);
        if (!exec_handle->engine->initialize(module_handle->module)) {
            last_error_message = "Failed to initialize JIT execution engine";
            PGX_ERROR("pgx_jit_create_execution_handle: %s", last_error_message.c_str());
            delete exec_handle;
            return nullptr;
        }

        if (!exec_handle->engine->setupMemoryContexts()) {
            last_error_message = "Failed to setup memory contexts";
            PGX_ERROR("pgx_jit_create_execution_handle: %s", last_error_message.c_str());
            delete exec_handle;
            return nullptr;
        }

        PGX_LOG(JIT, DEBUG, "JIT execution handle created successfully");
        return exec_handle;

    } catch (const std::exception& e) {
        last_error_message = std::string("Exception creating JIT handle: ") + e.what();
        PGX_ERROR("pgx_jit_create_execution_handle: %s", last_error_message.c_str());
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception creating JIT handle";
        PGX_ERROR("pgx_jit_create_execution_handle: %s", last_error_message.c_str());
        return nullptr;
    }
}

int pgx_jit_execute_query(ExecutionHandle* exec_handle, void* estate, void* dest) {
    if (!exec_handle) {
        last_error_message = "Null execution handle";
        PGX_ERROR("pgx_jit_execute_query: %s", last_error_message.c_str());
        return -1;
    }

    if (!estate || !dest) {
        last_error_message = "Null estate or destination receiver";
        PGX_ERROR("pgx_jit_execute_query: %s", last_error_message.c_str());
        return -2;
    }

    try {
        bool success = exec_handle->engine->executeCompiledQuery(estate, dest);

        if (!success) {
            last_error_message = "JIT query execution failed";
            PGX_ERROR("pgx_jit_execute_query: %s", last_error_message.c_str());
            return -3;
        }

        PGX_LOG(JIT, DEBUG, "JIT query execution completed successfully");
        return 0;

    } catch (const std::exception& e) {
        last_error_message = std::string("Exception during JIT execution: ") + e.what();
        PGX_ERROR("pgx_jit_execute_query: %s", last_error_message.c_str());
        return -4;
    } catch (...) {
        last_error_message = "Unknown exception during JIT execution";
        PGX_ERROR("pgx_jit_execute_query: %s", last_error_message.c_str());
        return -5;
    }
}

void pgx_jit_destroy_execution_handle(ExecutionHandle* exec_handle) {
    if (exec_handle) {
        PGX_LOG(JIT, DEBUG, "Destroying JIT execution handle");
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

// Forward declarations for PostgreSQL types
struct PlannedStmt;
struct EState;
struct _DestReceiver;
typedef struct _DestReceiver DestReceiver;

namespace mlir_runner {

// High-level JIT execution function that handles the complete process
bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
    auto moduleHandle = pgx_jit_create_module_handle(&module);
    if (!moduleHandle) {
        PGX_ERROR("Failed to create module handle for JIT execution");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: %s", error);
        }
        return false;
    }

    auto execHandle = pgx_jit_create_execution_handle(moduleHandle);
    pgx_jit_destroy_module_handle(moduleHandle);

    if (!execHandle) {
        PGX_ERROR("Failed to create JIT execution handle");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: %s", error);
        }
        return false;
    }

    int result = pgx_jit_execute_query(execHandle, estate, dest);
    pgx_jit_destroy_execution_handle(execHandle);

    if (result != 0) {
        PGX_ERROR("JIT query execution failed with code: %d", result);
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: %s", error);
        }
        return false;
    }

    return true;
}

} // namespace mlir_runner