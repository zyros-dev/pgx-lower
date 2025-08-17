#include "execution/jit_execution_engine.h"
#include "execution/logging.h"
#include "runtime/tuple_access.h"
#include <fstream>
#include <dlfcn.h>
#include <filesystem>
#include <array>
#include <cstdlib>
#include <chrono>

// MLIR includes
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// LLVM includes
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

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
void* rt_get_execution_context();
bool add_tuple_to_result(int64_t value);
void mark_results_ready_for_streaming();
void prepare_computed_results(int32_t numColumns);
}

namespace pgx_lower {
namespace execution {

// Forward declaration for destructor
class WrappedExecutionEngine;

// WrappedExecutionEngine class following LingoDB pattern from lingo-db/lib/runner/runner.cpp:551-634
class WrappedExecutionEngine {
    std::unique_ptr<mlir::ExecutionEngine> engine;
    size_t jitTime;
    void* mainFuncPtr;
    bool useStaticLinking;
    
private:
    // Helper method to create module translation lambda
    static auto createModuleTranslationLambda() {
        return [](::mlir::Operation* op, llvm::LLVMContext& context) 
            -> std::unique_ptr<llvm::Module> {
            
            PGX_DEBUG("WrappedExecutionEngine: Translating MLIR module to LLVM IR");
            if (!op) {
                PGX_ERROR("Null operation in module translation");
                return nullptr;
            }
            
            auto module = mlir::cast<::mlir::ModuleOp>(op);
            
            // Register translations before conversion
            mlir::registerLLVMDialectTranslation(*module->getContext());
            
            auto llvmModule = mlir::translateModuleToLLVMIR(module, context, "PostgreSQLJITModule");
            
            if (!llvmModule) {
                PGX_ERROR("Failed to translate MLIR to LLVM IR");
                return nullptr;
            }
            
            // Fix function visibility for dlsym lookup (critical for dlopen approach)
            for (auto& func : llvmModule->functions()) {
                std::string funcName = func.getName().str();
                if (funcName == "main" || funcName == "_mlir_ciface_main") {
                    func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                    func.setVisibility(llvm::GlobalValue::DefaultVisibility);
                    func.setDSOLocal(true);
                    PGX_INFO("Set " + funcName + " to ExternalLinkage for dlsym");
                }
            }
            
            return llvmModule;
        };
    }
    
    // Helper method to create optimization lambda
    static auto createOptimizationLambda(llvm::CodeGenOptLevel optLevel) {
        return [optLevel](llvm::Module* module) -> llvm::Error {
            PGX_DEBUG("WrappedExecutionEngine: Optimizing LLVM module");
            
            if (optLevel == llvm::CodeGenOptLevel::None) {
                PGX_INFO("Skipping optimization (level = None)");
                return llvm::Error::success();
            }
            
            llvm::legacy::FunctionPassManager funcPM(module);
            funcPM.add(llvm::createInstructionCombiningPass());
            funcPM.add(llvm::createReassociatePass());
            funcPM.add(llvm::createGVNPass());
            funcPM.add(llvm::createCFGSimplificationPass());
            
            funcPM.doInitialization();
            for (auto& func : *module) {
                if (!func.hasOptNone()) {
                    funcPM.run(func);
                }
            }
            funcPM.doFinalization();
            
            return llvm::Error::success();
        };
    }
    
    // Helper method to initialize the engine with options
    void initializeEngineWithOptions(mlir::ModuleOp module, llvm::CodeGenOptLevel optLevel) {
        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.llvmModuleBuilder = createModuleTranslationLambda();
        engineOptions.transformer = createOptimizationLambda(optLevel);
        engineOptions.jitCodeGenOptLevel = optLevel;
        engineOptions.enableObjectDump = true;  // Enable for debugging
        
        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        if (!maybeEngine) {
            PGX_ERROR("Failed to create ExecutionEngine");
            return;
        }
        
        engine = std::move(*maybeEngine);
        
        // Try standard lookup first (unless forcing static)
        if (!useStaticLinking) {
            auto lookupResult = engine->lookup("main");
            if (!lookupResult) {
                lookupResult = engine->lookup("_mlir_ciface_main");
            }
            
            if (lookupResult) {
                mainFuncPtr = lookupResult.get();
                PGX_INFO("Found main function via ExecutionEngine at: " + 
                         std::to_string(reinterpret_cast<uintptr_t>(mainFuncPtr)));
            } else {
                PGX_WARNING("Main function not found via ExecutionEngine::lookup, will use static linking");
                useStaticLinking = true;
            }
        }
    }
    
public:
    WrappedExecutionEngine(mlir::ModuleOp module, llvm::CodeGenOptLevel optLevel, bool forceStatic = false) 
        : mainFuncPtr(nullptr), useStaticLinking(forceStatic) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Initialize the execution engine with proper options
        initializeEngineWithOptions(module, optLevel);
        
        auto end = std::chrono::high_resolution_clock::now();
        jitTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        PGX_INFO("JIT compilation took: " + std::to_string(jitTime / 1000.0) + " ms");
    }
    
    bool succeeded() const {
        return mainFuncPtr != nullptr || useStaticLinking;
    }
    
    // Helper method to compile object file to shared library
    bool compileObjectToSharedLibrary(const std::string& objectFile, const std::string& sharedLibFile) {
        std::string cmd = "g++ -shared -fPIC -o " + sharedLibFile + " " + objectFile + " 2>&1";
        PGX_INFO("Compiling: " + cmd);
        
        auto* pPipe = ::popen(cmd.c_str(), "r");
        if (!pPipe) {
            PGX_ERROR("Failed to compile JIT object to shared library");
            return false;
        }
        
        std::array<char, 256> buffer;
        std::string result;
        while (!std::feof(pPipe)) {
            auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
            result.append(buffer.data(), bytes);
        }
        
        auto rc = ::pclose(pPipe);
        if (WEXITSTATUS(rc)) {
            PGX_ERROR("Compilation failed: " + result);
            return false;
        }
        
        PGX_INFO("Successfully compiled to " + sharedLibFile);
        return true;
    }
    
    // Helper method to load shared library
    void* loadSharedLibrary(const std::string& path) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        
        if (!handle) {
            const char* dlError = dlerror();
            PGX_ERROR("Cannot load shared library: " + std::string(dlError ? dlError : "unknown error"));
            return nullptr;
        }
        
        PGX_INFO("Successfully loaded shared library");
        return handle;
    }
    
    // Helper method to lookup main function in loaded library
    void* lookupMainFunction(void* handle) {
        void* funcPtr = dlsym(handle, "main");
        if (!funcPtr) {
            funcPtr = dlsym(handle, "_mlir_ciface_main");
        }
        
        if (!funcPtr) {
            const char* dlError = dlerror();
            PGX_ERROR("Cannot find main function via dlsym: " + std::string(dlError ? dlError : "unknown"));
            return nullptr;
        }
        
        PGX_INFO("Successfully found main function via dlsym at: " + 
                 std::to_string(reinterpret_cast<uintptr_t>(funcPtr)));
        return funcPtr;
    }
    
    bool linkStatic() {
        // Following LingoDB pattern from runner.cpp:584-624
        PGX_INFO("Using LingoDB static linking pattern with dlopen/dlsym");
        
        auto currPath = std::filesystem::current_path();
        
        // Dump JIT object to file
        engine->dumpToObjectFile("pgx_jit_module.o");
        PGX_INFO("Dumped JIT object to pgx_jit_module.o");
        
        // Compile to shared library
        if (!compileObjectToSharedLibrary("pgx_jit_module.o", "pgx_jit_module.so")) {
            return false;
        }
        
        // Load shared library
        std::string soPath = currPath.string() + "/pgx_jit_module.so";
        void* handle = loadSharedLibrary(soPath);
        if (!handle) {
            return false;
        }
        
        // Lookup main function
        mainFuncPtr = lookupMainFunction(handle);
        return mainFuncPtr != nullptr;
    }
    
    size_t getJitTime() const {
        return jitTime;
    }
    
    void* getMainFuncPtr() const {
        return mainFuncPtr;
    }
    
    mlir::ExecutionEngine* getEngine() {
        return engine.get();
    }
};

// Destructor to properly delete WrappedExecutionEngine
PostgreSQLJITExecutionEngine::~PostgreSQLJITExecutionEngine() {
    if (wrappedEngine) {
        delete static_cast<WrappedExecutionEngine*>(wrappedEngine);
        wrappedEngine = nullptr;
    }
}

void PostgreSQLJITExecutionEngine::configureLLVMTargetMachine() {
    PGX_DEBUG("Configuring LLVM target machine");
    
    static bool llvm_initialized = false;
    if (!llvm_initialized) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        llvm_initialized = true;
        PGX_DEBUG("LLVM target initialization completed");
    }
    
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    PGX_INFO("LLVM target triple: " + targetTriple);
    
    std::string errorMessage;
    const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
    if (!target) {
        PGX_ERROR("Failed to lookup LLVM target: " + errorMessage);
    }
}

bool PostgreSQLJITExecutionEngine::validateModuleForCompilation(::mlir::ModuleOp module) {
    PGX_DEBUG("Validating MLIR module for compilation");
    
    if (!module) {
        PGX_ERROR("Module is null");
        return false;
    }
    
    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Module verification failed");
        return false;
    }
    
    auto mainFunc = module.lookupSymbol<::mlir::LLVM::LLVMFuncOp>("main");
    if (!mainFunc) {
        PGX_WARNING("Module does not contain 'main' function");
    } else {
        PGX_DEBUG("Found main function");
    }
    
    return true;
}

bool PostgreSQLJITExecutionEngine::setupJITOptimizationPipeline() {
    PGX_DEBUG("Setting up JIT optimization pipeline");
    PGX_INFO("JIT optimization level set to: " + std::to_string(static_cast<int>(optimizationLevel)));
    return true;
}

bool PostgreSQLJITExecutionEngine::compileToLLVMIR(::mlir::ModuleOp module) {
    PGX_DEBUG("Compiling MLIR module to LLVM IR");
    
    if (!validateModuleForCompilation(module)) {
        PGX_ERROR("Module validation failed");
        return false;
    }
    
    return true;
}

// Helper method to register dialect translations for MLIR compilation
void PostgreSQLJITExecutionEngine::registerDialectTranslations(::mlir::ModuleOp module) {
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    
    module->getContext()->appendDialectRegistry(registry);
    mlir::registerLLVMDialectTranslation(*module->getContext());
}

// Helper method to create and initialize the wrapped execution engine
bool PostgreSQLJITExecutionEngine::createWrappedExecutionEngine(::mlir::ModuleOp module) {
    auto* wrapped = new WrappedExecutionEngine(module, optimizationLevel);
    wrappedEngine = wrapped;
    
    if (!wrapped->succeeded()) {
        PGX_WARNING("Initial lookup failed, trying static linking");
        if (!wrapped->linkStatic()) {
            PGX_ERROR("Failed to initialize JIT execution engine");
            delete wrapped;
            wrappedEngine = nullptr;
            return false;
        }
    }
    
    // Register runtime functions if engine is available
    if (wrapped->getEngine()) {
        registerPostgreSQLRuntimeFunctions();
    }
    
    return true;
}

bool PostgreSQLJITExecutionEngine::initialize(::mlir::ModuleOp module) {
    PGX_INFO("Initializing PostgreSQL JIT execution engine with LingoDB pattern");
    
    if (initialized) {
        PGX_WARNING("Execution engine already initialized");
        return true;
    }
    
    // Validate module
    if (!validateModuleForCompilation(module)) {
        return false;
    }
    
    // Register dialect translations
    registerDialectTranslations(module);
    
    // Configure LLVM target
    configureLLVMTargetMachine();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Create and initialize the wrapped execution engine
    if (!createWrappedExecutionEngine(module)) {
        return false;
    }
    
    initialized = true;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    PGX_INFO("JIT execution engine initialized in " + std::to_string(duration / 1000.0) + " ms");
    
    return true;
}

void PostgreSQLJITExecutionEngine::registerLingoDRuntimeContextFunctions() {
    PGX_DEBUG("Registering LingoDB runtime context functions");
    
    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    if (!wrapped || !wrapped->getEngine()) {
        PGX_ERROR("Cannot register functions - engine not initialized");
        return;
    }
    
    wrapped->getEngine()->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            // rt_set_execution_context function
            symbolMap[interner("rt_set_execution_context")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(
                    +[](void* context_ptr) -> void {
                        static thread_local void* global_execution_context = nullptr;
                        global_execution_context = context_ptr;
                        PGX_DEBUG("Set execution context");
                    }
                )),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            // rt_get_execution_context function
            symbolMap[interner("rt_get_execution_context")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(
                    +[]() -> void* {
                        static thread_local void* global_execution_context = nullptr;
                        return global_execution_context;
                    }
                )),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

// Helper method to get list of runtime table builder functions
std::vector<std::string> getTableBuilderFunctions() {
    return {
        "rt_tablebuilder_create",
        "rt_tablebuilder_build", 
        "rt_tablebuilder_nextrow",
        "rt_tablebuilder_addint64",
        "rt_tablebuilder_destroy"
    };
}

// Helper method to get list of data source iteration functions
std::vector<std::string> getDataSourceIterationFunctions() {
    return {
        "rt_datasourceiteration_start",
        "rt_datasourceiteration_isvalid",
        "rt_datasourceiteration_next",
        "rt_datasourceiteration_access",
        "rt_datasourceiteration_end"
    };
}

// Helper method to get list of context management functions
std::vector<std::string> getContextManagementFunctions() {
    return {
        "rt_get_execution_context"
    };
}

// Helper lambda to create the symbol registration function
auto createSymbolRegistrationLambda() {
    return [](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        
        void* handle = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            PGX_ERROR("Failed to open current process for symbol lookup");
            return symbolMap;
        }
        
        auto registerFunction = [&](const std::string& funcName) {
            void* funcPtr = dlsym(handle, funcName.c_str());
            if (funcPtr) {
                symbolMap[interner(funcName)] = {
                    llvm::orc::ExecutorAddr::fromPtr(funcPtr),
                    llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
                };
                PGX_DEBUG("Registered: " + funcName);
            }
        };
        
        // Register all runtime function groups
        auto tableBuilderFuncs = getTableBuilderFunctions();
        auto dataSourceFuncs = getDataSourceIterationFunctions();
        auto contextFuncs = getContextManagementFunctions();
        
        for (const auto& funcName : tableBuilderFuncs) {
            registerFunction(funcName);
        }
        for (const auto& funcName : dataSourceFuncs) {
            registerFunction(funcName);
        }
        for (const auto& funcName : contextFuncs) {
            registerFunction(funcName);
        }
        
        dlclose(handle);
        
        PGX_INFO("Registered " + std::to_string(symbolMap.size()) + " runtime functions");
        
        return symbolMap;
    };
}

void PostgreSQLJITExecutionEngine::registerCRuntimeFunctions() {
    PGX_INFO("Registering C runtime functions");
    
    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    if (!wrapped || !wrapped->getEngine()) {
        PGX_ERROR("Cannot register functions - engine not initialized");
        return;
    }
    
    wrapped->getEngine()->registerSymbols(createSymbolRegistrationLambda());
}

void PostgreSQLJITExecutionEngine::registerPostgreSQLRuntimeFunctions() {
    PGX_INFO("Registering PostgreSQL runtime functions");
    
    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    if (!wrapped || !wrapped->getEngine()) {
        PGX_ERROR("Cannot register runtime functions - execution engine not initialized");
        return;
    }
    
    registerCRuntimeFunctions();
    registerLingoDRuntimeContextFunctions();
}

bool PostgreSQLJITExecutionEngine::setupMemoryContexts() {
    PGX_DEBUG("Setting up PostgreSQL memory contexts");
    
#ifdef POSTGRESQL_EXTENSION
    if (!CurrentMemoryContext) {
        PGX_ERROR("No current memory context available");
        return false;
    }
    
    PG_TRY();
    {
        if (!CurTransactionContext) {
            PGX_ERROR("No current transaction context available");
            return false;
        }
        
        MemoryContext oldcontext = MemoryContextSwitchTo(CurTransactionContext);
        
        if (CurrentMemoryContext != CurTransactionContext) {
            MemoryContextSwitchTo(oldcontext);
            PGX_ERROR("Failed to switch to transaction memory context");
            return false;
        }
        
        MemoryContextSwitchTo(oldcontext);
        
        PGX_DEBUG("Memory contexts configured successfully");
        return true;
    }
    PG_CATCH();
    {
        PGX_ERROR("Memory context switch failed");
        return false;
    }
    PG_END_TRY();
#else
    PGX_DEBUG("Running in unit test environment");
    return true;
#endif
}

// Helper method to lookup the execution function from wrapped engine
void* PostgreSQLJITExecutionEngine::lookupExecutionFunction(WrappedExecutionEngine* wrapped) {
    void* funcPtr = wrapped->getMainFuncPtr();
    if (!funcPtr) {
        PGX_WARNING("No cached function pointer, trying static linking");
        if (!wrapped->linkStatic()) {
            PGX_ERROR("Static linking failed");
            return nullptr;
        }
        funcPtr = wrapped->getMainFuncPtr();
    }
    
    if (!funcPtr) {
        PGX_ERROR("Main function pointer is null");
        return nullptr;
    }
    
    PGX_INFO("Executing function at: " + std::to_string(reinterpret_cast<uintptr_t>(funcPtr)));
    return funcPtr;
}

// Helper method to invoke the compiled function with proper error handling
bool PostgreSQLJITExecutionEngine::invokeCompiledFunction(void* funcPtr, void* estate, void* dest) {
#ifdef POSTGRESQL_EXTENSION
    bool executionSuccess = false;
    PG_TRY();
    {
        typedef void (*query_func)();
        auto fn = (query_func) funcPtr;
        fn();
        executionSuccess = true;
        PGX_INFO("JIT function execution completed");
    }
    PG_CATCH();
    {
        PGX_ERROR("PostgreSQL exception during JIT execution");
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    return executionSuccess;
#else
    try {
        typedef void (*query_func)();
        auto fn = (query_func) funcPtr;
        fn();
        PGX_INFO("JIT function execution completed");
        return true;
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during JIT execution: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Unknown exception during JIT execution");
        return false;
    }
#endif
}

// Helper method to log execution timing metrics
void PostgreSQLJITExecutionEngine::logExecutionMetrics(std::chrono::microseconds duration) {
    PGX_INFO("JIT execution took " + std::to_string(duration.count() / 1000.0) + " ms");
    PGX_INFO("JIT execution completed successfully");
}

bool PostgreSQLJITExecutionEngine::executeCompiledQuery(void* estate, void* dest) {
    PGX_INFO("Executing JIT compiled query using LingoDB pattern");
    
    if (!initialized || !wrappedEngine) {
        PGX_ERROR("JIT engine not initialized");
        return false;
    }
    
    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    
    if (!estate || !dest) {
        PGX_ERROR("Null estate or dest receiver");
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Lookup the execution function
    void* funcPtr = lookupExecutionFunction(wrapped);
    if (!funcPtr) {
        return false;
    }
    
    // Invoke the compiled function
    bool success = invokeCompiledFunction(funcPtr, estate, dest);
    if (!success) {
        return false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // Log execution metrics
    logExecutionMetrics(duration);
    
    return true;
}

} // namespace execution
} // namespace pgx_lower