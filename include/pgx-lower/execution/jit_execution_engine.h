#pragma once

#include <memory>
#include <string>
#include <chrono>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/Module.h"

namespace pgx_lower {
namespace execution {

class WrappedExecutionEngine;

class PostgreSQLJITExecutionEngine {
public:
    PostgreSQLJITExecutionEngine();
    ~PostgreSQLJITExecutionEngine();
    
    PostgreSQLJITExecutionEngine(const PostgreSQLJITExecutionEngine&) = delete;
    PostgreSQLJITExecutionEngine& operator=(const PostgreSQLJITExecutionEngine&) = delete;
    PostgreSQLJITExecutionEngine(PostgreSQLJITExecutionEngine&&) = default;
    PostgreSQLJITExecutionEngine& operator=(PostgreSQLJITExecutionEngine&&) = default;
    
    bool initialize(::mlir::ModuleOp module);
    bool setupJITOptimizationPipeline();
    bool compileToLLVMIR(::mlir::ModuleOp module);
    bool isInitialized() const;
    void setOptimizationLevel(llvm::CodeGenOptLevel level);
    bool setupMemoryContexts();
    bool executeCompiledQuery(void* estate, void* dest);
    
private:
    // Implementation details hidden in .cpp file using anonymous namespaces
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace execution
} // namespace pgx_lower