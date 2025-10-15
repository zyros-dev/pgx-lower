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
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

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

namespace llvm { class Module; }
namespace mlir_runner {
    extern void dumpLLVMIR(llvm::Module* module, const std::string& title, pgx_lower::log::Category phase);
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
    
    void registerDialectTranslations(::mlir::ModuleOp module);
    bool createWrappedExecutionEngine(::mlir::ModuleOp module);
    
    void* lookupExecutionFunction(WrappedExecutionEngine* wrapped);
    bool invokeCompiledFunction(void* funcPtr, void* estate, void* dest);
    void logExecutionMetrics(std::chrono::microseconds duration);
};

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
            
            // Count input Standard MLIR operations
            size_t inputOpsCount = 0;
            module.walk([&](::mlir::Operation* operation) {
                inputOpsCount++;
            });
            
            PGX_LOG(JIT, IO, "MLIR→LLVM IN: Standard MLIR Module with %zu operations", inputOpsCount);
            
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
                PGX_ERROR("LLVM module verification failed: %s", verifyError.c_str());
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

            size_t outputFuncCount = llvmModule->size();
            size_t outputInstCount = 0;
            for (const auto& func : *llvmModule) {
                for (const auto& bb : func) {
                    outputInstCount += bb.size();
                }
            }
            
            PGX_LOG(JIT, IO, "MLIR→LLVM OUT: LLVM IR Module with %zu functions, %zu instructions", outputFuncCount, outputInstCount);

            return llvmModule;
        };
    }

    static auto createOptimizationLambda(llvm::CodeGenOptLevel optLevel) {
        PGX_LOG(JIT, DEBUG, "Creating optimization lambda!");
        return [optLevel](llvm::Module* module) -> llvm::Error {
            PGX_LOG(JIT, DEBUG, "Running optimization lambda!");
            if (optLevel == llvm::CodeGenOptLevel::None) {
                return llvm::Error::success();
            }

            try {
                PGX_LOG(JIT, DEBUG, "Creating PassBuilder and analysis managers");

                PGX_LOG(JIT, DEBUG, "Skipping LLVM optimization passes (modern PM needs debugging)");
                mlir_runner::dumpLLVMIR(module, "LLVM IR WITHOUT OPTIMIZATION PASSES", log::Category::JIT);
                return llvm::Error::success();

                // TODO: Fix modern pass manager crash
                // Modern pass manager setup with default tuning options
                llvm::PipelineTuningOptions PTO;
                PTO.LoopUnrolling = false;
                PTO.LoopVectorization = false;
                PTO.SLPVectorization = false;

                llvm::PassBuilder PB(nullptr, PTO);
                llvm::LoopAnalysisManager LAM;
                llvm::FunctionAnalysisManager FAM;
                llvm::CGSCCAnalysisManager CGAM;
                llvm::ModuleAnalysisManager MAM;

                PGX_LOG(JIT, DEBUG, "Registering loop analyses");
                PB.registerLoopAnalyses(LAM);

                PGX_LOG(JIT, DEBUG, "Registering function analyses");
                PB.registerFunctionAnalyses(FAM);

                PGX_LOG(JIT, DEBUG, "Registering CGSCC analyses");
                PB.registerCGSCCAnalyses(CGAM);

                PGX_LOG(JIT, DEBUG, "Registering module analyses");
                PB.registerModuleAnalyses(MAM);

                PGX_LOG(JIT, DEBUG, "Cross-registering proxies");
                PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

                PGX_LOG(JIT, DEBUG, "Building function pass pipeline");
                // Build function pass pipeline
                // Key ordering: InstCombine before PromotePass enables better alloca hoisting
                llvm::FunctionPassManager FPM;
                FPM.addPass(llvm::InstCombinePass());
                FPM.addPass(llvm::PromotePass());  // mem2reg in modern PM
                FPM.addPass(llvm::InstCombinePass());
                FPM.addPass(llvm::createFunctionToLoopPassAdaptor(llvm::LICMPass(llvm::LICMOptions())));
                FPM.addPass(llvm::ReassociatePass());
                FPM.addPass(llvm::GVNPass());
                FPM.addPass(llvm::SimplifyCFGPass());

                PGX_LOG(JIT, DEBUG, "Running passes on %zu functions", module->size());
                for (auto& func : *module) {
                    if (func.isDeclaration()) {
                        PGX_LOG(JIT, DEBUG, "Skipping external function: %s", func.getName().str().c_str());
                        continue;
                    }
                    if (!func.hasOptNone()) {
                        PGX_LOG(JIT, DEBUG, "Running passes on function: %s", func.getName().str().c_str());
                        FPM.run(func, FAM);
                    }
                }

                PGX_LOG(JIT, DEBUG, "Optimization passes completed successfully");
                mlir_runner::dumpLLVMIR(module, "LLVM IR AFTER OPTIMIZATION PASSES", log::Category::JIT);

                return llvm::Error::success();
            } catch (const std::exception& e) {
                PGX_ERROR("Exception in optimization lambda: %s", e.what());
                return llvm::make_error<llvm::StringError>(
                    "Optimization failed: " + std::string(e.what()),
                    llvm::inconvertibleErrorCode()
                );
            } catch (...) {
                PGX_ERROR("Unknown exception in optimization lambda");
                return llvm::make_error<llvm::StringError>(
                    "Optimization failed with unknown exception",
                    llvm::inconvertibleErrorCode()
                );
            }
        };
    }

    void initializeEngineWithOptions(mlir::ModuleOp module, llvm::CodeGenOptLevel optLevel) {
        PGX_LOG(JIT, DEBUG, "running initializeEngineWithOptions");
        auto moduleBuilder = createModuleTranslationLambda();
        auto transformer = createOptimizationLambda(optLevel);

        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.llvmModuleBuilder = moduleBuilder;
        engineOptions.transformer = transformer;
        engineOptions.jitCodeGenOptLevel = optLevel;
        engineOptions.enableObjectDump = true;

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
        PGX_LOG(JIT, DEBUG, "init wrapped execution ending");
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
        std::string cmd = "g++ -shared -fPIC -Wl,--unresolved-symbols=ignore-all -o " + sharedLibFile + " " + objectFile + " 2>&1";

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
            PGX_ERROR("Compilation failed: %s", result.c_str());
            return false;
        }

        return true;
    }

    void* loadSharedLibrary(const std::string& path) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);

        if (!handle) {
            const char* dlError = dlerror();
            PGX_ERROR("Cannot load shared library: %s", dlError ? dlError : "unknown error");
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
            PGX_ERROR("Cannot find main function via dlsym: %s", dlError ? dlError : "unknown");
            return nullptr;
        }

        return funcPtr;
    }

    bool linkStatic() {
        std::string tmpDir = "/tmp";
        std::string objectFile = tmpDir + "/pgx_jit_module.o";
        std::string sharedLibFile = tmpDir + "/pgx_jit_module.so";

        engine->dumpToObjectFile(objectFile);

        if (!compileObjectToSharedLibrary(objectFile, sharedLibFile)) {
            return false;
        }

        std::string soPath = sharedLibFile;
        void* handle = loadSharedLibrary(soPath);
        if (!handle) {
            return false;
        }

        mainFuncPtr = lookupMainFunction(handle);

        setContextPtr = dlsym(handle, "rt_set_execution_context");
        if (!setContextPtr) {
            const char* dlError = dlerror();
            PGX_WARNING("Cannot find rt_set_execution_context via dlsym: %s", dlError ? dlError : "unknown");
        }
        else {
            PGX_LOG(JIT, DEBUG, "Found rt_set_execution_context via dlsym at: %p", setContextPtr);
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
    PGX_LOG(JIT, DEBUG, "LLVM target triple: %s", targetTriple.c_str());

    std::string errorMessage;
    const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
    if (!target) {
        PGX_ERROR("Failed to lookup LLVM target: %s", errorMessage.c_str());
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
    PGX_LOG(JIT, IO, "JIT Initialize OUT: ExecutionEngine ready (compilation took %.2f ms)", duration / 1000.0);

    return true;
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

    PGX_LOG(JIT, DEBUG, "Executing function at: %p", funcPtr);
    return funcPtr;
}

bool PostgreSQLJITExecutionEngine::Impl::invokeCompiledFunction(void* funcPtr, void* estate, void* dest) {
#ifdef POSTGRESQL_EXTENSION
    bool executionSuccess = false;

    MemoryContext savedContext = CurrentMemoryContext;

    PG_TRY();
    {
        if (wrappedEngine) {
            auto* wrapped = static_cast<WrappedExecutionEngine*>(wrappedEngine);
            void* setContextPtr = wrapped->getSetContextPtr();
            if (setContextPtr) {
                PGX_LOG(JIT, DEBUG, "Setting execution context via rt_set_execution_context");
                PGX_LOG(JIT, DEBUG, "estate pointer value: %p", estate);
                typedef void (*set_context_func)(void*);
                auto setCtx = (set_context_func)setContextPtr;
                setCtx(estate);
                PGX_LOG(JIT, DEBUG, "Execution context set successfully");
            }
        }

        typedef void (*query_func)();
        auto fn = (query_func)funcPtr;
        PGX_LOG(JIT, DEBUG, "About to execute JIT-compiled function at address %p", funcPtr);

        // Check slot state before execution
        if (g_tuple_streamer.slot) {
            PGX_LOG(JIT, DEBUG, "Before JIT execution: slot=%p, tts_nvalid=%d, tts_tupleDescriptor=%p",
                    g_tuple_streamer.slot, g_tuple_streamer.slot->tts_nvalid,
                    g_tuple_streamer.slot->tts_tupleDescriptor);
        }

        if (!fn) {
            PGX_ERROR("FATAL: JIT function pointer is NULL!");
            return false;
        }
        fn();
        PGX_LOG(JIT, DEBUG, "JIT function call returned successfully");

        // Check slot state after execution
        if (g_tuple_streamer.slot) {
            PGX_LOG(JIT, DEBUG, "After JIT execution: slot=%p, tts_nvalid=%d, tts_tupleDescriptor=%p",
                    g_tuple_streamer.slot, g_tuple_streamer.slot->tts_nvalid,
                    g_tuple_streamer.slot->tts_tupleDescriptor);
        }

        executionSuccess = true;
        PGX_LOG(JIT, DEBUG, "JIT function execution completed");

        // PostgreSQL still needs access to JIT-allocated data.
        // and PostgreSQL has finished processing all tuples.

        CHECK_FOR_INTERRUPTS();
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(savedContext);

        auto* edata = CopyErrorData();
        PGX_ERROR("PostgreSQL exception during JIT execution: %s (SQLSTATE: %s, detail: %s, hint: %s)",
                  edata->message ? edata->message : "<no message>",
                  edata->sqlerrcode ? unpack_sql_state(edata->sqlerrcode) : "<no code>",
                  edata->detail ? edata->detail : "<no detail>",
                  edata->hint ? edata->hint : "<no hint>");
        FreeErrorData(edata);

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
        PGX_LOG(JIT, DEBUG, "JIT function execution completed");
        return true;
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during JIT execution: %s", e.what());
        return false;
    } catch (...) {
        PGX_ERROR("Unknown exception during JIT execution");
        return false;
    }
#endif
}

void PostgreSQLJITExecutionEngine::Impl::logExecutionMetrics(std::chrono::microseconds duration) {
    PGX_LOG(JIT, DEBUG, "JIT execution took %.3f ms", duration.count() / 1000.0);
    PGX_LOG(JIT, DEBUG, "JIT execution completed successfully");
}

bool PostgreSQLJITExecutionEngine::Impl::executeCompiledQuery(void* estate, void* dest) {
    PGX_LOG(JIT, IO, "JIT Execute IN: CompiledQuery (estate=%p, dest=%p)", estate, dest);

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

    logExecutionMetrics(duration);
    PGX_LOG(JIT, IO, "JIT Execute OUT: Query completed successfully (%.2f ms)", duration.count() / 1000.0);

    return true;
}

}} // namespace pgx_lower::execution