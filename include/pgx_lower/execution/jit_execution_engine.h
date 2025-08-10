#pragma once

#include <memory>
#include <string>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/Module.h"

namespace pgx_lower {
namespace execution {

/**
 * @brief PostgreSQL-compatible JIT execution engine for MLIR modules
 * 
 * This class provides the infrastructure for compiling MLIR modules to LLVM IR
 * and preparing them for JIT execution within PostgreSQL. It follows LingoDB
 * patterns but adapts them for PostgreSQL's memory and execution contexts.
 */
class PostgreSQLJITExecutionEngine {
private:
    std::unique_ptr<mlir::ExecutionEngine> engine;
    bool initialized = false;
    
    // JIT optimization configuration
    llvm::CodeGenOptLevel optimizationLevel = llvm::CodeGenOptLevel::Default;
    
    // Module validation and preparation
    bool validateModuleForCompilation(::mlir::ModuleOp module);
    void configureLLVMTargetMachine();
    
    // Runtime function registration helpers
    void registerDSARuntimeFunctions();
    void registerPostgreSQLSPIFunctions();
    void registerMemoryManagementFunctions();
    void registerDataSourceFunctions();
    void registerRuntimeSupportFunctions();
    
public:
    PostgreSQLJITExecutionEngine() = default;
    ~PostgreSQLJITExecutionEngine() = default;
    
    // Non-copyable, movable
    PostgreSQLJITExecutionEngine(const PostgreSQLJITExecutionEngine&) = delete;
    PostgreSQLJITExecutionEngine& operator=(const PostgreSQLJITExecutionEngine&) = delete;
    PostgreSQLJITExecutionEngine(PostgreSQLJITExecutionEngine&&) = default;
    PostgreSQLJITExecutionEngine& operator=(PostgreSQLJITExecutionEngine&&) = default;
    
    /**
     * @brief Initialize the execution engine with an MLIR module
     * @param module The MLIR module to compile
     * @return true if initialization succeeded, false otherwise
     */
    bool initialize(::mlir::ModuleOp module);
    
    /**
     * @brief Setup JIT optimization pipeline configuration
     * @return true if setup succeeded, false otherwise
     */
    bool setupJITOptimizationPipeline();
    
    /**
     * @brief Compile the loaded MLIR module to LLVM IR
     * @param module The MLIR module to compile
     * @return true if compilation succeeded, false otherwise
     */
    bool compileToLLVMIR(::mlir::ModuleOp module);
    
    /**
     * @brief Check if the engine is initialized and ready
     * @return true if initialized, false otherwise
     */
    bool isInitialized() const { return initialized; }
    
    /**
     * @brief Set the JIT optimization level
     * @param level LLVM optimization level
     */
    void setOptimizationLevel(llvm::CodeGenOptLevel level) { 
        optimizationLevel = level; 
    }
    
    /**
     * @brief Register PostgreSQL runtime functions with the JIT symbol table
     * 
     * This method registers all runtime functions needed for query execution:
     * - DSA runtime functions (table builder, iteration)
     * - PostgreSQL SPI functions (table access, tuple operations)
     * - PostgreSQL memory management (palloc, pfree)
     * - Utility functions (ereport, elog)
     */
    void registerPostgreSQLRuntimeFunctions();
    
    /**
     * @brief Setup PostgreSQL memory contexts for JIT execution
     * @return true if setup succeeded, false otherwise
     */
    bool setupMemoryContexts();
    
    /**
     * @brief Execute the compiled query using JIT
     * @param estate PostgreSQL execution state
     * @param dest PostgreSQL destination receiver for results
     * @return true if execution succeeded, false otherwise
     */
    bool executeCompiledQuery(void* estate, void* dest);
};

} // namespace execution
} // namespace pgx_lower