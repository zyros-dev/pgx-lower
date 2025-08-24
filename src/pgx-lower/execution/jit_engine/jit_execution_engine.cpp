#include "pgx-lower/execution/jit_execution_engine.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <filesystem>
#include <array>
#include <cstdlib>
#include <chrono>

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
#include "miscadmin.h"
}
#endif

extern "C" {
void* rt_get_execution_context();
bool add_tuple_to_result(int64_t value);
void mark_results_ready_for_streaming();
void prepare_computed_results(int32_t numColumns);
}

namespace pgx_lower { namespace execution {

class WrappedExecutionEngine;

// Impl class for PostgreSQLJITExecutionEngine (Pimpl pattern)
class PostgreSQLJITExecutionEngine::Impl {
public:
    Impl() : wrappedEngine(nullptr), initialized(false), 
             optimizationLevel(llvm::CodeGenOptLevel::None) {}
    ~Impl();
    
    // Public interface methods
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
    
private:
    void* wrappedEngine;
    bool initialized;
    llvm::CodeGenOptLevel optimizationLevel;
    
    // Private implementation methods
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
};

// WrappedExecutionEngine class following LingoDB pattern from lingo-db/lib/runner/runner.cpp:551-634
class WrappedExecutionEngine {
    std::unique_ptr<mlir::ExecutionEngine> engine;
    size_t jitTime;
    void* mainFuncPtr;
    void* setContextPtr;
    bool useStaticLinking;

   private:
    static auto createModuleTranslationLambda() {
        return [](::mlir::Operation* op, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
            if (!op) {
                PGX_ERROR("Null operation in module translation");
                return nullptr;
            }

            auto module = mlir::cast<::mlir::ModuleOp>(op);
            mlir::registerLLVMDialectTranslation(*module->getContext());
            auto llvmModule = mlir::translateModuleToLLVMIR(module, context, "PostgreSQLJITModule");
            if (!llvmModule) {
                PGX_ERROR("Failed to translate MLIR to LLVM IR");
                return nullptr;
            }

            std::string verifyError;
            llvm::raw_string_ostream verifyStream(verifyError);
            if (llvm::verifyModule(*llvmModule, &verifyStream)) {
                verifyStream.flush();
                PGX_ERROR("LLVM module verification failed: " + verifyError);
                return nullptr;
            }

            for (auto& func : llvmModule->functions()) {
                std::string funcName = func.getName().str();
                if (funcName == "main" || funcName == "_mlir_ciface_main") {
                    func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                    func.setVisibility(llvm::GlobalValue::DefaultVisibility);
                    func.setDSOLocal(true);
                }
            }

            return llvmModule;
        };
    }

    static auto createOptimizationLambda(llvm::CodeGenOptLevel optLevel) {
        return [optLevel](llvm::Module* module) -> llvm::Error {
            if (optLevel == llvm::CodeGenOptLevel::None) {
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

    void initializeEngineWithOptions(mlir::ModuleOp module, llvm::CodeGenOptLevel optLevel) {
        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.llvmModuleBuilder = createModuleTranslationLambda();
        engineOptions.transformer = createOptimizationLambda(optLevel);
        engineOptions.jitCodeGenOptLevel = optLevel;
        engineOptions.enableObjectDump = true; // Enable for debugging

        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        if (!maybeEngine) {
            PGX_ERROR("Failed to create ExecutionEngine");
            return;
        }

        engine = std::move(*maybeEngine);

        if (!useStaticLinking) {
            auto lookupResult = engine->lookup("main");
            if (!lookupResult) {
                lookupResult = engine->lookup("_mlir_ciface_main");
            }

            if (lookupResult) {
                mainFuncPtr = lookupResult.get();
            }
            else {
                PGX_WARNING("Main function not found via ExecutionEngine::lookup, will use static linking");
                useStaticLinking = true;
            }

            // Lookup setContext function per LingoDB pattern
            auto setContextResult = engine->lookup("rt_set_execution_context");
            if (setContextResult) {
                setContextPtr = setContextResult.get();
            }
            else {
                PGX_WARNING("rt_set_execution_context function not found - context management disabled");
            }
        }
    }

   public:
    WrappedExecutionEngine(mlir::ModuleOp module, llvm::CodeGenOptLevel optLevel, bool forceStatic = false)
    : mainFuncPtr(nullptr)
    , setContextPtr(nullptr)
    , useStaticLinking(forceStatic) {
        auto start = std::chrono::high_resolution_clock::now();
        initializeEngineWithOptions(module, optLevel);

        auto end = std::chrono::high_resolution_clock::now();
        jitTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    bool succeeded() const {
        // Per LingoDB pattern: both mainFuncPtr and setContextPtr must be valid
        return (mainFuncPtr != nullptr && setContextPtr != nullptr) || useStaticLinking;
    }

    void* getSetContextPtr() const { return setContextPtr; }

    bool compileObjectToSharedLibrary(const std::string& objectFile, const std::string& sharedLibFile) {
        // Link with undefined symbols allowed - they'll be resolved from the main extension
        std::string cmd =
            "g++ -shared -fPIC -Wl,--unresolved-symbols=ignore-all -o " + sharedLibFile + " " + objectFile + " 2>&1";

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

        return true;
    }

    void* loadSharedLibrary(const std::string& path) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);

        if (!handle) {
            const char* dlError = dlerror();
            PGX_ERROR("Cannot load shared library: " + std::string(dlError ? dlError : "unknown error"));
            return nullptr;
        }

        return handle;
    }

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

        return funcPtr;
    }

    bool linkStatic() {
        auto currPath = std::filesystem::current_path();

        // Dump JIT object to file
        engine->dumpToObjectFile("pgx_jit_module.o");

        // Compile to shared library
        if (!compileObjectToSharedLibrary("pgx_jit_module.o", "pgx_jit_module.so")) {
            return false;
        }

        std::string soPath = currPath.string() + "/pgx_jit_module.so";
        void* handle = loadSharedLibrary(soPath);
        if (!handle) {
            return false;
        }

        // Lookup main function
        mainFuncPtr = lookupMainFunction(handle);

        // Lookup setContext function per LingoDB pattern
        setContextPtr = dlsym(handle, "rt_set_execution_context");
        if (!setContextPtr) {
            const char* dlError = dlerror();
            PGX_WARNING("Cannot find rt_set_execution_context via dlsym: " + std::string(dlError ? dlError : "unknown"));
        }
        else {
            PGX_INFO("Found rt_set_execution_context via dlsym at: "
                     + std::to_string(reinterpret_cast<uintptr_t>(setContextPtr)));
        }

        // Both functions must be found for success per LingoDB pattern
        return mainFuncPtr != nullptr && setContextPtr != nullptr;
    }

    size_t getJitTime() const { return jitTime; }

    void* getMainFuncPtr() const { return mainFuncPtr; }

    mlir::ExecutionEngine* getEngine() { return engine.get(); }
};

// Destructor to properly delete WrappedExecutionEngine
// Constructor and destructor for main class (Pimpl pattern)
PostgreSQLJITExecutionEngine::PostgreSQLJITExecutionEngine() 
    : pImpl(std::make_unique<Impl>()) {}

PostgreSQLJITExecutionEngine::~PostgreSQLJITExecutionEngine() = default;

// Forward all public methods to Impl
bool PostgreSQLJITExecutionEngine::initialize(::mlir::ModuleOp module) {
    return pImpl->initialize(module);
}

bool PostgreSQLJITExecutionEngine::setupJITOptimizationPipeline() {
    return pImpl->setupJITOptimizationPipeline();
}

bool PostgreSQLJITExecutionEngine::compileToLLVMIR(::mlir::ModuleOp module) {
    return pImpl->compileToLLVMIR(module);
}

bool PostgreSQLJITExecutionEngine::isInitialized() const {
    return pImpl->isInitialized();
}

void PostgreSQLJITExecutionEngine::setOptimizationLevel(llvm::CodeGenOptLevel level) {
    pImpl->setOptimizationLevel(level);
}

void PostgreSQLJITExecutionEngine::registerPostgreSQLRuntimeFunctions() {
    pImpl->registerPostgreSQLRuntimeFunctions();
}

bool PostgreSQLJITExecutionEngine::setupMemoryContexts() {
    return pImpl->setupMemoryContexts();
}

bool PostgreSQLJITExecutionEngine::executeCompiledQuery(void* estate, void* dest) {
    return pImpl->executeCompiledQuery(estate, dest);
}

// Impl destructor
PostgreSQLJITExecutionEngine::Impl::~Impl() {
    if (wrappedEngine) {
        delete static_cast<WrappedExecutionEngine*>(wrappedEngine);
        wrappedEngine = nullptr;
    }
}

void PostgreSQLJITExecutionEngine::Impl::configureLLVMTargetMachine() {
    static bool llvm_initialized = false;
    if (!llvm_initialized) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        llvm_initialized = true;
    }

    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    PGX_INFO("LLVM target triple: " + targetTriple);

    std::string errorMessage;
    const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
    if (!target) {
        PGX_ERROR("Failed to lookup LLVM target: " + errorMessage);
    }
}

bool PostgreSQLJITExecutionEngine::Impl::validateModuleForCompilation(::mlir::ModuleOp module) {
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
    }
    else {
    }

    return true;
}

bool PostgreSQLJITExecutionEngine::Impl::setupJITOptimizationPipeline() {
    return true;
}

bool PostgreSQLJITExecutionEngine::Impl::compileToLLVMIR(::mlir::ModuleOp module) {
    if (!validateModuleForCompilation(module)) {
        PGX_ERROR("Module validation failed");
        return false;
    }

    return true;
}

void PostgreSQLJITExecutionEngine::Impl::registerDialectTranslations(::mlir::ModuleOp module) {
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);

    module->getContext()->appendDialectRegistry(registry);
    mlir::registerLLVMDialectTranslation(*module->getContext());
}

bool PostgreSQLJITExecutionEngine::Impl::createWrappedExecutionEngine(::mlir::ModuleOp module) {
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

    if (wrapped->getEngine()) {
        registerPostgreSQLRuntimeFunctions();
    }

    return true;
}

bool PostgreSQLJITExecutionEngine::Impl::initialize(::mlir::ModuleOp module) {
    if (initialized) {
        PGX_WARNING("Execution engine already initialized");
        return true;
    }

    if (!validateModuleForCompilation(module)) {
        return false;
    }

    registerDialectTranslations(module);

    configureLLVMTargetMachine();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (!createWrappedExecutionEngine(module)) {
        return false;
    }

    initialized = true;

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    return true;
}

void PostgreSQLJITExecutionEngine::Impl::registerLingoDRuntimeContextFunctions() {
    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    if (!wrapped || !wrapped->getEngine()) {
        PGX_ERROR("Cannot register functions - engine not initialized");
        return;
    }

    wrapped->getEngine()->registerSymbols([](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;

        // rt_set_execution_context function
        symbolMap[interner("rt_set_execution_context")] = {
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(+[](void* context_ptr) -> void {
                static thread_local void* global_execution_context = nullptr;
                global_execution_context = context_ptr;
            })),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // rt_get_execution_context function
        symbolMap[interner("rt_get_execution_context")] = {
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(+[]() -> void* {
                static thread_local void* global_execution_context = nullptr;
                return global_execution_context;
            })),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        return symbolMap;
    });
}

std::vector<std::string> getTableBuilderFunctions() {
    return {"rt_tablebuilder_create",
            "rt_tablebuilder_build",
            "rt_tablebuilder_nextrow",
            "rt_tablebuilder_addint64",
            "rt_tablebuilder_addint32",
            "rt_tablebuilder_addbool",
            "rt_tablebuilder_destroy"};
}

std::vector<std::string> getDataSourceIterationFunctions() {
    return {"rt_datasourceiteration_start",
            "rt_datasourceiteration_isvalid",
            "rt_datasourceiteration_next",
            "rt_datasourceiteration_access",
            "rt_datasourceiteration_end"};
}

std::vector<std::string> getContextManagementFunctions() {
    return {"rt_get_execution_context"};
}

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
                symbolMap[interner(funcName)] = {llvm::orc::ExecutorAddr::fromPtr(funcPtr),
                                                 llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
            }
        };

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

void PostgreSQLJITExecutionEngine::Impl::registerCRuntimeFunctions() {
    PGX_INFO("Registering C runtime functions");

    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    if (!wrapped || !wrapped->getEngine()) {
        PGX_ERROR("Cannot register functions - engine not initialized");
        return;
    }

    wrapped->getEngine()->registerSymbols(createSymbolRegistrationLambda());
}

void PostgreSQLJITExecutionEngine::Impl::registerPostgreSQLRuntimeFunctions() {
    PGX_INFO("Registering PostgreSQL runtime functions");

    auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
    if (!wrapped || !wrapped->getEngine()) {
        PGX_ERROR("Cannot register runtime functions - execution engine not initialized");
        return;
    }

    registerCRuntimeFunctions();
    registerLingoDRuntimeContextFunctions();
}

bool PostgreSQLJITExecutionEngine::Impl::setupMemoryContexts() {
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

        return true;
    }
    PG_CATCH();
    {
        PGX_ERROR("Memory context switch failed");
        return false;
    }
    PG_END_TRY();
#else
    return true;
#endif
}

void* PostgreSQLJITExecutionEngine::Impl::lookupExecutionFunction(WrappedExecutionEngine* wrapped) {
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

bool PostgreSQLJITExecutionEngine::Impl::invokeCompiledFunction(void* funcPtr, void* estate, void* dest) {
#ifdef POSTGRESQL_EXTENSION
    bool executionSuccess = false;

    MemoryContext savedContext = CurrentMemoryContext;

    // MEMORY FIX: Do NOT switch contexts before JIT execution
    // Switching here can cause issues with tuple streaming

    PG_TRY();
    {
        if (wrappedEngine) {
            auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
            void* setContextPtr = wrapped->getSetContextPtr();
            if (setContextPtr) {
                PGX_INFO("Setting execution context via rt_set_execution_context");
                PGX_INFO("estate pointer value: " + std::to_string(reinterpret_cast<uintptr_t>(estate)));
                typedef void (*set_context_func)(void*);
                auto setCtx = (set_context_func)setContextPtr;
                setCtx(estate);
                PGX_INFO("Execution context set successfully");
            }
        }

        typedef void (*query_func)();
        auto fn = (query_func)funcPtr;
        PGX_INFO("About to execute JIT-compiled function");
        fn();
        executionSuccess = true;
        PGX_INFO("JIT function execution completed");

        // PostgreSQL still needs access to JIT-allocated data.
        // and PostgreSQL has finished processing all tuples.

        CHECK_FOR_INTERRUPTS();
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(savedContext);
        PGX_ERROR("PostgreSQL exception during JIT execution");
        PG_RE_THROW();
    }
    PG_END_TRY();

    // MEMORY FIX: Restore context AFTER all tuple processing is complete
    MemoryContextSwitchTo(savedContext);

    return executionSuccess;
#else
    try {
        typedef void (*query_func)();
        auto fn = (query_func)funcPtr;
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

void PostgreSQLJITExecutionEngine::Impl::logExecutionMetrics(std::chrono::microseconds duration) {
    PGX_INFO("JIT execution took " + std::to_string(duration.count() / 1000.0) + " ms");
    PGX_INFO("JIT execution completed successfully");
}

bool PostgreSQLJITExecutionEngine::Impl::executeCompiledQuery(void* estate, void* dest) {
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

}} // namespace pgx_lower::execution