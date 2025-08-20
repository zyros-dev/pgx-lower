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
private:
    void* wrappedEngine = nullptr;
    bool initialized = false;
    
    llvm::CodeGenOptLevel optimizationLevel = llvm::CodeGenOptLevel::None;
    
    bool validateModuleForCompilation(::mlir::ModuleOp module);
    void configureLLVMTargetMachine();
    
    void registerDSARuntimeFunctions();
    void registerPostgreSQLSPIFunctions();
    void registerMemoryManagementFunctions();
    void registerDataSourceFunctions();
    void registerRuntimeSupportFunctions();
    void registerLingoDRuntimeContextFunctions();
    void registerCRuntimeFunctions();
    
    void registerDialectTranslations(::mlir::ModuleOp module);
    bool createWrappedExecutionEngine(::mlir::ModuleOp module);
    
    void* lookupExecutionFunction(WrappedExecutionEngine* wrapped);
    bool invokeCompiledFunction(void* funcPtr, void* estate, void* dest);
    void logExecutionMetrics(std::chrono::microseconds duration);
    
public:
    PostgreSQLJITExecutionEngine() = default;
    ~PostgreSQLJITExecutionEngine();
    
    PostgreSQLJITExecutionEngine(const PostgreSQLJITExecutionEngine&) = delete;
    PostgreSQLJITExecutionEngine& operator=(const PostgreSQLJITExecutionEngine&) = delete;
    PostgreSQLJITExecutionEngine(PostgreSQLJITExecutionEngine&&) = default;
    PostgreSQLJITExecutionEngine& operator=(PostgreSQLJITExecutionEngine&&) = default;
    
    bool initialize(::mlir::ModuleOp module);
    bool setupJITOptimizationPipeline();
    bool compileToLLVMIR(::mlir::ModuleOp module);
    bool isInitialized() const { return initialized; }
    void setOptimizationLevel(llvm::CodeGenOptLevel level) { 
        optimizationLevel = level; 
    }
    void registerPostgreSQLRuntimeFunctions();
    bool setupMemoryContexts();
    bool executeCompiledQuery(void* estate, void* dest);
};

} // namespace execution
} // namespace pgx_lower