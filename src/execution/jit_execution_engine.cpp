#include "execution/jit_execution_engine.h"
#include "execution/logging.h"

// MLIR includes
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// LLVM includes
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include <chrono>

// PostgreSQL headers for runtime function declarations
#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/elog.h"
}
#endif

// Forward declarations of runtime functions
extern "C" {
// DSA Runtime Functions (from Phase 4e-3)
void* pgx_runtime_create_table_builder(const char* schema);
void pgx_runtime_append_i64(void* builder, size_t col_idx, int64_t value);
void pgx_runtime_append_i64_direct(void* builder, int64_t value);
void pgx_runtime_append_nullable_i64(void* builder, bool is_null, int64_t value);
void pgx_runtime_append_null(void* builder, size_t col_idx);
void pgx_runtime_table_next_row(void* builder);

// PostgreSQL SPI Functions (from Phase 4d) - implemented in tuple_access.cpp
void* pg_table_open(const char* table_name);
int64_t pg_get_next_tuple(void* table_handle);
int32_t pg_extract_field(void* tuple, int32_t field_index);
void pg_store_result(void* result);
void pg_store_result_i32(int32_t value);
void pg_store_result_i64(int64_t value);
void pg_store_result_f64(double value);
void pg_store_result_text(const char* value);

// PostgreSQL tuple access functions (from tuple_access.cpp)
void* open_postgres_table(const char* tableName);
int64_t read_next_tuple_from_table(void* tableHandle);
void close_postgres_table(void* tableHandle);
int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null);
int64_t get_text_field(void* tuple_handle, int32_t field_index, bool* is_null);
double get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null);
int32_t get_int_field_mlir(int64_t iteration_signal, int32_t field_index);
void store_int_result(int32_t columnIndex, int32_t value, bool isNull);
void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
void store_text_result(int32_t columnIndex, const char* value, bool isNull);
void store_field_as_datum(int32_t columnIndex, int64_t iteration_signal, int32_t field_index);
bool add_tuple_to_result(int64_t value);
void mark_results_ready_for_streaming();
void prepare_computed_results(int32_t numColumns);

// PostgreSQL runtime functions (from postgresql_runtime.cpp)
void* pgx_exec_alloc_state_raw(int64_t size);
void pgx_exec_free_state(void* state);
void pgx_exec_set_tuple_count(void* exec_context, int64_t count);
int64_t pgx_exec_get_tuple_count(void* exec_context);
void* pgx_threadlocal_create(int64_t size);
void* pgx_threadlocal_get(void* tls);
void pgx_threadlocal_merge(void* dest, void* src);
void* pgx_datasource_get(void* table_ref);
void* pgx_datasource_iteration_init(void* datasource, int64_t start, int64_t end);
int8_t pgx_datasource_iteration_iterate(void* iteration, void** row_out);
void* pgx_buffer_create_zeroed(int64_t size);
void* pgx_buffer_iterate(void* buffer, int64_t index);
void* pgx_growing_buffer_create(int64_t initial_capacity);
void pgx_growing_buffer_insert(void* buffer, void* value, int64_t value_size);
}

namespace pgx_lower {
namespace execution {

void PostgreSQLJITExecutionEngine::configureLLVMTargetMachine() {
    PGX_DEBUG("Configuring LLVM target machine");
    
    // Initialize LLVM native target for JIT compilation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    
    // Get the target triple for the current platform
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    PGX_INFO("LLVM target triple: " + targetTriple);
    
    // Verify target is available
    std::string errorMessage;
    const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
    if (!target) {
        PGX_ERROR("Failed to lookup LLVM target: " + errorMessage);
        return;
    }
    
    PGX_DEBUG("LLVM target machine configured successfully");
}

bool PostgreSQLJITExecutionEngine::validateModuleForCompilation(::mlir::ModuleOp module) {
    PGX_DEBUG("Validating MLIR module for compilation");
    
    if (!module) {
        PGX_ERROR("Module is null");
        return false;
    }
    
    // Verify the module is well-formed
    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Module verification failed");
        return false;
    }
    
    // Check for main function (required for execution)
    auto mainFunc = module.lookupSymbol<::mlir::func::FuncOp>("main");
    if (!mainFunc) {
        PGX_WARNING("Module does not contain 'main' function - execution may not be possible");
    } else {
        PGX_DEBUG("Found main function with " + std::to_string(mainFunc.getNumArguments()) + 
                  " arguments and " + std::to_string(mainFunc.getNumResults()) + " results");
    }
    
    PGX_DEBUG("Module validation successful");
    return true;
}

bool PostgreSQLJITExecutionEngine::setupJITOptimizationPipeline() {
    PGX_DEBUG("Setting up JIT optimization pipeline");
    
    // Configuration is stored internally and will be used during engine creation
    // The actual pipeline setup happens in initialize() when creating the engine
    
    PGX_INFO("JIT optimization level set to: " + std::to_string(static_cast<int>(optimizationLevel)));
    
    return true;
}

bool PostgreSQLJITExecutionEngine::compileToLLVMIR(::mlir::ModuleOp module) {
    PGX_DEBUG("Compiling MLIR module to LLVM IR");
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (!validateModuleForCompilation(module)) {
        PGX_ERROR("Module validation failed, cannot compile");
        return false;
    }
    
    // The actual compilation happens during engine initialization
    // This method validates that the module is ready for compilation
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    PGX_INFO("Module compilation preparation took " + std::to_string(duration / 1000.0) + " ms");
    
    return true;
}

bool PostgreSQLJITExecutionEngine::initialize(::mlir::ModuleOp module) {
    PGX_DEBUG("Initializing PostgreSQL JIT execution engine");
    
    if (initialized) {
        PGX_WARNING("Execution engine already initialized");
        return true;
    }
    
    // Validate module first
    if (!validateModuleForCompilation(module)) {
        return false;
    }
    
    // Register all standard dialect translations before any LLVM operations
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    module->getContext()->appendDialectRegistry(registry);
    
    // Configure LLVM target machine
    configureLLVMTargetMachine();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Define the module translation function (MLIR -> LLVM IR)
    auto moduleTranslation = [](::mlir::Operation* op, llvm::LLVMContext& context) 
        -> std::unique_ptr<llvm::Module> {
        auto module = mlir::cast<::mlir::ModuleOp>(op);
        PGX_DEBUG("Translating MLIR module to LLVM IR");
        
        // Perform the translation
        auto llvmModule = mlir::translateModuleToLLVMIR(module, context, "PostgreSQLJITModule");
        
        if (!llvmModule) {
            PGX_ERROR("Failed to translate MLIR module to LLVM IR");
            return nullptr;
        }
        
        PGX_DEBUG("Successfully translated to LLVM IR");
        return llvmModule;
    };
    
    // Define the optimization function
    auto optimizeModule = [this](llvm::Module* module) -> llvm::Error {
        PGX_DEBUG("Optimizing LLVM module");
        
        if (optimizationLevel == llvm::CodeGenOptLevel::None) {
            PGX_INFO("Skipping LLVM optimization (optimization level = None)");
            return llvm::Error::success();
        }
        
        // Create and run optimization passes
        llvm::legacy::FunctionPassManager funcPM(module);
        
        // Add standard optimization passes based on level
        if (optimizationLevel != llvm::CodeGenOptLevel::None) {
            funcPM.add(llvm::createInstructionCombiningPass());
            funcPM.add(llvm::createReassociatePass());
            funcPM.add(llvm::createGVNPass());
            funcPM.add(llvm::createCFGSimplificationPass());
        }
        
        funcPM.doInitialization();
        for (auto& func : *module) {
            if (!func.hasOptNone()) {
                funcPM.run(func);
            }
        }
        funcPM.doFinalization();
        
        PGX_DEBUG("LLVM optimization complete");
        return llvm::Error::success();
    };
    
    // Create the execution engine with our configuration
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.llvmModuleBuilder = moduleTranslation;
    engineOptions.transformer = optimizeModule;
    engineOptions.jitCodeGenOptLevel = optimizationLevel;
    engineOptions.enableObjectDump = false; // Disable for PostgreSQL compatibility
    
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    
    if (!maybeEngine) {
        auto error = maybeEngine.takeError();
        std::string errorStr;
        llvm::raw_string_ostream errorStream(errorStr);
        errorStream << error;
        PGX_ERROR("Failed to create execution engine: " + errorStream.str());
        return false;
    }
    
    engine = std::move(*maybeEngine);
    initialized = true;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    PGX_INFO("JIT execution engine initialized in " + std::to_string(duration / 1000.0) + " ms");
    
    return true;
}

void PostgreSQLJITExecutionEngine::registerDSARuntimeFunctions() {
    PGX_DEBUG("Registering DSA runtime functions");
    
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            // Table builder functions
            symbolMap[interner("pgx_runtime_create_table_builder")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_runtime_create_table_builder)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_runtime_append_i64")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_runtime_append_i64)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_runtime_append_i64_direct")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_runtime_append_i64_direct)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_runtime_append_nullable_i64")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_runtime_append_nullable_i64)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_runtime_append_null")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_runtime_append_null)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_runtime_table_next_row")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_runtime_table_next_row)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerPostgreSQLSPIFunctions() {
    PGX_DEBUG("Registering PostgreSQL SPI functions");
    
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            symbolMap[interner("pg_table_open")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_table_open)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_get_next_tuple")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_get_next_tuple)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_extract_field")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_extract_field)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_store_result")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_store_result)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_store_result_i32")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_store_result_i32)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_store_result_i64")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_store_result_i64)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_store_result_f64")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_store_result_f64)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pg_store_result_text")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pg_store_result_text)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerMemoryManagementFunctions() {
    PGX_DEBUG("Registering memory management functions");
    
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            // Memory allocation
            symbolMap[interner("pgx_exec_alloc_state_raw")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_exec_alloc_state_raw)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_exec_free_state")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_exec_free_state)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            // Execution context
            symbolMap[interner("pgx_exec_set_tuple_count")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_exec_set_tuple_count)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_exec_get_tuple_count")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_exec_get_tuple_count)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            // Thread-local storage (PostgreSQL doesn't use threads, but required for compatibility)
            symbolMap[interner("pgx_threadlocal_create")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_threadlocal_create)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_threadlocal_get")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_threadlocal_get)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_threadlocal_merge")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_threadlocal_merge)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerDataSourceFunctions() {
    PGX_DEBUG("Registering data source and buffer operations");
    
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            // Data source
            symbolMap[interner("pgx_datasource_get")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_datasource_get)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_datasource_iteration_init")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_datasource_iteration_init)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_datasource_iteration_iterate")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_datasource_iteration_iterate)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            // Buffers
            symbolMap[interner("pgx_buffer_create_zeroed")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_buffer_create_zeroed)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_buffer_iterate")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_buffer_iterate)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_growing_buffer_create")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_growing_buffer_create)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("pgx_growing_buffer_insert")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(pgx_growing_buffer_insert)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerRuntimeSupportFunctions() {
    PGX_DEBUG("Registering runtime support functions");
    
    // PostgreSQL tuple access functions
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            symbolMap[interner("open_postgres_table")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(open_postgres_table)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("read_next_tuple_from_table")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(read_next_tuple_from_table)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("close_postgres_table")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(close_postgres_table)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("get_int_field")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(get_int_field)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("get_int_field_mlir")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(get_int_field_mlir)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("get_text_field")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(get_text_field)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("get_numeric_field")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(get_numeric_field)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
    
    // Result storage functions
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            symbolMap[interner("store_int_result")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(store_int_result)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("store_bigint_result")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(store_bigint_result)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("store_text_result")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(store_text_result)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("store_field_as_datum")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(store_field_as_datum)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("add_tuple_to_result")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(add_tuple_to_result)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("mark_results_ready_for_streaming")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(mark_results_ready_for_streaming)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            symbolMap[interner("prepare_computed_results")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(prepare_computed_results)),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerPostgreSQLRuntimeFunctions() {
    PGX_DEBUG("Registering PostgreSQL runtime functions with JIT symbol table");
    
    if (!engine) {
        PGX_ERROR("Cannot register runtime functions - execution engine not initialized");
        return;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Register all function groups
    registerDSARuntimeFunctions();
    registerPostgreSQLSPIFunctions();
    registerMemoryManagementFunctions();
    registerDataSourceFunctions();
    registerRuntimeSupportFunctions();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    PGX_INFO("Registered all PostgreSQL runtime functions in " + std::to_string(duration / 1000.0) + " ms");
}

bool PostgreSQLJITExecutionEngine::setupMemoryContexts() {
    PGX_DEBUG("Setting up PostgreSQL memory contexts for JIT execution");
    
#ifdef POSTGRESQL_EXTENSION
    // In PostgreSQL, memory contexts are already set up by the backend
    // We just need to ensure we're in the right context for JIT operations
    
    if (!CurrentMemoryContext) {
        PGX_ERROR("No current memory context available");
        return false;
    }
    
    // Use PostgreSQL's error handling mechanism for memory context operations
    PG_TRY();
    {
        // Validate we have a transaction context
        if (!CurTransactionContext) {
            PGX_ERROR("No current transaction context available");
            return false;
        }
        
        // Switch to a safe memory context for JIT operations
        MemoryContext oldcontext = MemoryContextSwitchTo(CurTransactionContext);
        
        // Validate the context switch succeeded
        if (CurrentMemoryContext != CurTransactionContext) {
            MemoryContextSwitchTo(oldcontext);
            PGX_ERROR("Failed to switch to transaction memory context");
            return false;
        }
        
        // Register any memory-context-specific functions if needed
        // For now, we just switch back
        MemoryContextSwitchTo(oldcontext);
        
        PGX_DEBUG("Memory contexts configured successfully");
        return true;
    }
    PG_CATCH();
    {
        // Handle PostgreSQL errors gracefully
        PGX_ERROR("Memory context switch failed - PostgreSQL error occurred");
        return false;
    }
    PG_END_TRY();
#else
    // In unit tests, we don't have PostgreSQL memory contexts
    PGX_DEBUG("Running in unit test environment - no PostgreSQL memory contexts to setup");
    return true;
#endif
}

bool PostgreSQLJITExecutionEngine::executeCompiledQuery(void* estate, void* dest) {
    PGX_DEBUG("Executing JIT compiled query");
    
    if (!initialized || !engine) {
        PGX_ERROR("Cannot execute query - JIT engine not initialized");
        return false;
    }
    
    if (!estate || !dest) {
        PGX_ERROR("Cannot execute query - null estate or dest receiver");
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Look up the compiled query function
    // The function should be named "compiled_query" or "main" depending on the lowering
    auto queryFuncResult = engine->lookup("compiled_query");
    if (!queryFuncResult) {
        PGX_WARNING("Failed to find 'compiled_query' function, trying 'main'");
        queryFuncResult = engine->lookup("main");
        
        if (!queryFuncResult) {
            PGX_ERROR("Failed to lookup compiled query function - neither 'compiled_query' nor 'main' found");
            return false;
        }
    }
    
    // Get the function pointer
    auto funcPtr = queryFuncResult.get();
    if (!funcPtr) {
        PGX_ERROR("Failed to get function pointer from JIT lookup result");
        return false;
    }
    
    PGX_INFO("Found JIT compiled query function, preparing for execution");
    
    // Cast to the expected function signature
    // The function signature should match what the MLIR lowering produces
    // For Test 1, this is likely: int compiled_query(void* estate, void* dest)
    typedef int (*QueryFunctionType)(void*, void*);
    auto compiledQuery = reinterpret_cast<QueryFunctionType>(funcPtr);
    
    // Execute the compiled query
    PGX_INFO("Executing JIT compiled query for Test 1");
    
    try {
        int result = compiledQuery(estate, dest);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        PGX_INFO("JIT query execution completed in " + std::to_string(duration / 1000.0) + " ms with result: " + std::to_string(result));
        
        // Return true if execution succeeded (result == 0)
        return (result == 0);
        
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during JIT query execution: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Unknown exception during JIT query execution");
        return false;
    }
}

} // namespace execution
} // namespace pgx_lower