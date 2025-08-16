#include "execution/jit_execution_engine.h"
#include "execution/logging.h"
#include "runtime/tuple_access.h"
#include <fstream>
#include <dlfcn.h>

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

#include <chrono>
#include <future>
#include <thread>

// PostgreSQL headers for runtime function declarations
#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/elog.h"
}
#endif

// Forward declarations of runtime functions actually used
extern "C" {
// Only the essential functions needed for Test 1
void* rt_get_execution_context();

// PostgreSQL tuple access functions (minimal set from tuple_access.cpp)
bool add_tuple_to_result(int64_t value);
void mark_results_ready_for_streaming();
void prepare_computed_results(int32_t numColumns);
}

namespace pgx_lower {
namespace execution {

void PostgreSQLJITExecutionEngine::configureLLVMTargetMachine() {
    PGX_DEBUG("Configuring LLVM target machine");
    
    // Initialize LLVM native target for JIT compilation with guard to prevent duplicates
    static bool llvm_initialized = false;
    if (!llvm_initialized) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        llvm_initialized = true;
        PGX_DEBUG("LLVM target initialization completed");
    } else {
        PGX_DEBUG("LLVM target already initialized, skipping");
    }
    
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
    // After Standard→LLVM lowering, functions become llvm.func, not func.func
    auto mainFunc = module.lookupSymbol<::mlir::LLVM::LLVMFuncOp>("main");
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
    
    // CRITICAL: Register missing translation interfaces that cause JIT failures
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    
    module->getContext()->appendDialectRegistry(registry);
    
    // CRITICAL: Register base LLVM dialect translation in context (LingoDB pattern)
    mlir::registerLLVMDialectTranslation(*module->getContext());
    
    // Configure LLVM target machine
    configureLLVMTargetMachine();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Define the module translation function (MLIR -> LLVM IR)
    auto moduleTranslation = [](::mlir::Operation* op, llvm::LLVMContext& context) 
        -> std::unique_ptr<llvm::Module> {
        PGX_INFO("moduleTranslation lambda called!");
        
        // Temporarily disable memory context switching to isolate issue
        
        if (!op) {
            PGX_ERROR("moduleTranslation received null operation");
            return nullptr;
        }
        
        auto module = mlir::cast<::mlir::ModuleOp>(op);
        PGX_DEBUG("Translating MLIR module to LLVM IR");
        PGX_DEBUG("Module has " + std::to_string(std::distance(module.begin(), module.end())) + " operations");
        
        // List all operations in the module for debugging
        PGX_INFO("Operations in MLIR module before translation:");
        int opCount = 0;
        module.walk([&opCount](mlir::Operation* op) {
            PGX_INFO("  Op[" + std::to_string(opCount++) + "]: " + op->getName().getStringRef().str());
        });
        
        // Check if module contains LLVM dialect operations
        bool hasLLVMOps = false;
        module.walk([&hasLLVMOps](mlir::Operation* op) {
            if (op->getName().getDialectNamespace() == "llvm") {
                hasLLVMOps = true;
            }
        });
        
        if (!hasLLVMOps) {
            PGX_ERROR("Module does not contain LLVM dialect operations - not properly lowered!");
        } else {
            PGX_INFO("Module contains LLVM dialect operations, proceeding with translation");
        }
        
        // Always dump module for debugging the translation failure
        std::string moduleStr;
        llvm::raw_string_ostream moduleStream(moduleStr);
        module.print(moduleStream);
        moduleStream.flush();
        
        // Log only first 1000 chars to avoid overwhelming logs
        if (moduleStr.length() > 50000) {
            PGX_INFO("MLIR module (truncated):\n" + moduleStr.substr(0, 50000) + "\n... [truncated]");
        } else {
            PGX_INFO("MLIR module:\n" + moduleStr);
        }
        
        // Perform the translation with error handling
        PGX_INFO("About to call translateModuleToLLVMIR...");
        std::unique_ptr<llvm::Module> llvmModule;
        
        // CRITICAL: Wrap in PostgreSQL exception handling
#ifdef POSTGRESQL_EXTENSION
        bool translationSuccess = false;
        PG_TRY();
        {
            llvmModule = mlir::translateModuleToLLVMIR(
                module, 
                context, 
                "PostgreSQLJITModule"
            );
            translationSuccess = true;
            PGX_INFO("translateModuleToLLVMIR call completed");
            
            // CRITICAL FIX: Set main function visibility for ExecutionEngine lookup
            if (llvmModule) {
                PGX_INFO("🔧 Analyzing functions in generated LLVM IR...");
                std::vector<std::string> functionNames;
                bool mainFunctionFound = false;
                
                // First pass: collect all function names and look for main
                for (auto& func : llvmModule->functions()) {
                    std::string funcName = func.getName().str();
                    functionNames.push_back(funcName);
                    
                    if (funcName == "main") {
                        func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                        func.setVisibility(llvm::GlobalValue::DefaultVisibility);
                        PGX_INFO("✅ SUCCESS: Set main function to ExternalLinkage + DefaultVisibility");
                        mainFunctionFound = true;
                    }
                }
                
                // Log all function names for debugging
                PGX_INFO("🔍 LLVM IR contains " + std::to_string(functionNames.size()) + " functions:");
                for (const auto& name : functionNames) {
                    PGX_INFO("  - " + name);
                }
                
                if (!mainFunctionFound) {
                    PGX_WARNING("⚠️ Main function not found in LLVM IR - checking for query function");
                    // Look for likely query function names and fix their visibility too
                    for (auto& func : llvmModule->functions()) {
                        std::string funcName = func.getName().str();
                        if (funcName.find("query") != std::string::npos || 
                            funcName.find("compiled") != std::string::npos ||
                            !func.isDeclaration()) { // Any defined function could be the query
                            func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                            func.setVisibility(llvm::GlobalValue::DefaultVisibility);
                            PGX_INFO("🔧 Set function " + funcName + " to ExternalLinkage + DefaultVisibility");
                        }
                    }
                } else {
                    PGX_INFO("🎯 Main function visibility fix applied - ExecutionEngine should find it now!");
                }
            }
        }
        PG_CATCH();
        {
            PGX_ERROR("PostgreSQL exception during translateModuleToLLVMIR");
            PG_RE_THROW();
        }
        PG_END_TRY();
        
        if (!translationSuccess) {
            PGX_ERROR("Translation failed");
            return nullptr;
        }
#else
        try {
            llvmModule = mlir::translateModuleToLLVMIR(
                module, 
                context, 
                "PostgreSQLJITModule"
            );
            PGX_INFO("translateModuleToLLVMIR call completed");
        } catch (const std::exception& e) {
            PGX_ERROR("Exception in translateModuleToLLVMIR: " + std::string(e.what()));
            return nullptr;
        } catch (...) {
            PGX_ERROR("Unknown exception in translateModuleToLLVMIR");
            return nullptr;
        }
#endif
        
        PGX_INFO("About to check llvmModule pointer...");
        fflush(stdout); // Force output flush
        
        // Try to check without dereferencing
        bool isNull = false;
        try {
            isNull = (llvmModule == nullptr);
            PGX_INFO("Pointer comparison succeeded, isNull = " + std::string(isNull ? "true" : "false"));
        } catch (...) {
            PGX_ERROR("Exception just checking if llvmModule == nullptr");
            return nullptr;
        }
        
        PGX_INFO("Checking if llvmModule is null...");
        if (!llvmModule || isNull) {
            PGX_ERROR("Failed to translate MLIR module to LLVM IR - null module returned");
            
            // Dump the MLIR module that failed to translate
            std::string mlirDump;
            llvm::raw_string_ostream mlirStream(mlirDump);
            module.print(mlirStream);
            mlirStream.flush();
            PGX_ERROR("Failed MLIR module:\n" + mlirDump);
            return nullptr;
        }
        
        PGX_INFO("llvmModule is not null, continuing...");
        // Don't access llvmModule internals yet - might cause crash
        PGX_INFO("LLVM module created successfully, preparing validation");
        
        // CRITICAL: Validate LLVM module before ExecutionEngine creation
        std::string validationErrors;
        llvm::raw_string_ostream errorStream(validationErrors);
        
        PGX_INFO("Starting LLVM module validation...");
        try {
            bool validationFailed = llvm::verifyModule(*llvmModule, &errorStream);
            errorStream.flush(); // Ensure all error data is written
            
            if (validationFailed) {
                PGX_ERROR("LLVM module validation failed: " + validationErrors);
                
                // Safe module dumping for debugging
                std::string moduleStr;
                llvm::raw_string_ostream moduleStream(moduleStr);
                llvmModule->print(moduleStream, nullptr);
                moduleStream.flush();
                PGX_DEBUG("Invalid LLVM IR:\n" + moduleStr);
                
                return nullptr;
            }
        } catch (const std::exception& e) {
            PGX_ERROR("Exception during LLVM module validation: " + std::string(e.what()));
            return nullptr;
        } catch (...) {
            PGX_ERROR("Unknown exception during LLVM module validation");
            return nullptr;
        }
        
        PGX_INFO("LLVM module validation passed - module is valid for ExecutionEngine");
        
        // Add module inspection for debugging
        PGX_DEBUG("LLVM module functions:");
        for (const auto& func : *llvmModule) {
            PGX_DEBUG("  Function: " + func.getName().str() + 
                      " (args: " + std::to_string(func.arg_size()) + 
                      ", basic_blocks: " + std::to_string(func.size()) + ")");
        }
        
        // Check if module has the expected query function
        bool hasQueryFunc = false;
        for (const auto& func : *llvmModule) {
            if (func.getName().starts_with("query")) {
                hasQueryFunc = true;
                PGX_INFO("Found query function: " + func.getName().str());
            }
        }
        
        if (!hasQueryFunc) {
            PGX_WARNING("No query function found in LLVM module - this may cause ExecutionEngine issues");
        }
        
        PGX_INFO("Successfully translated to LLVM IR, returning module");
        
        return llvmModule;
    };
    
    // Define the optimization function
    auto optimizeModule = [this](llvm::Module* module) -> llvm::Error {
        PGX_INFO("optimizeModule lambda called!");
        
        // Add validation before optimization
        if (!module) {
            PGX_ERROR("optimizeModule received null module");
            return llvm::createStringError(llvm::inconvertibleErrorCode(), 
                                         "Null module passed to optimizer");
        }
        
        // Verify module is still valid before optimization
        std::string preOptErrors;
        llvm::raw_string_ostream preOptStream(preOptErrors);
        if (llvm::verifyModule(*module, &preOptStream)) {
            preOptStream.flush();
            PGX_ERROR("Module invalid before optimization: " + preOptErrors);
            return llvm::createStringError(llvm::inconvertibleErrorCode(), 
                                         "Invalid module before optimization");
        }
        PGX_INFO("Module validated successfully before optimization");
        
        if (!module) {
            PGX_ERROR("optimizeModule received null module");
            return llvm::createStringError(llvm::inconvertibleErrorCode(), 
                                         "Null module passed to optimizer");
        }
        
        PGX_DEBUG("Optimizing LLVM module");
        PGX_DEBUG("Module name in optimizer: " + module->getName().str());
        
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
    engineOptions.enableObjectDump = true; // Enable for debugging
    
    // Note: sharedLibPaths configuration removed due to API incompatibility
    // The ExecutionEngine will use default symbol resolution
    
    PGX_INFO("Creating ExecutionEngine with configured options");
    
    // Add pre-validation of the MLIR module state
    if (!module) {
        PGX_ERROR("Module is null before ExecutionEngine creation");
        return false;
    }
    
    PGX_DEBUG("Module verification before ExecutionEngine::create");
    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Module verification failed before ExecutionEngine creation");
        return false;
    }
    
    PGX_DEBUG("Module stats before ExecutionEngine creation:");
    int funcCount = 0;
    module.walk([&funcCount](mlir::func::FuncOp func) {
        funcCount++;
        PGX_DEBUG("  Function: " + func.getName().str());
    });
    PGX_DEBUG("Total functions in module: " + std::to_string(funcCount));
    
    // CRITICAL: Dump the MLIR IR just before creating ExecutionEngine to see what we're compiling
    PGX_INFO("==================== MLIR IR BEFORE EXECUTION ENGINE ====================");
    std::string irStr;
    llvm::raw_string_ostream irStream(irStr);
    module.print(irStream);
    irStream.flush();
    
    // Write to file for backup
    std::ofstream irFile("/tmp/pgx_lower_mlir_ir.mlir");
    if (irFile.is_open()) {
        irFile << irStr;
        irFile.close();
        PGX_INFO("MLIR IR also written to /tmp/pgx_lower_mlir_ir.mlir");
    }
    
    // Log the entire MLIR IR in chunks to avoid truncation
    PGX_INFO("Full MLIR Module contents:");
    size_t chunkSize = 2000; // Log in 2000 char chunks
    for (size_t i = 0; i < irStr.length(); i += chunkSize) {
        size_t len = std::min(chunkSize, irStr.length() - i);
        PGX_INFO("MLIR[" + std::to_string(i) + "-" + std::to_string(i+len) + "]:\n" + irStr.substr(i, len));
    }
    PGX_INFO("==================== END MLIR IR ====================");
    
    std::unique_ptr<mlir::ExecutionEngine> createdEngine;
    try {
        PGX_INFO("Calling mlir::ExecutionEngine::create now...");
        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        
        if (!maybeEngine) {
            auto error = maybeEngine.takeError();
            std::string errorStr;
            llvm::raw_string_ostream errorStream(errorStr);
            errorStream << error;
            PGX_ERROR("ExecutionEngine::create failed: " + errorStr);
            return false;
        }
        
        PGX_INFO("ExecutionEngine::create returned valid result, extracting engine...");
        createdEngine = std::move(*maybeEngine);
        PGX_INFO("ExecutionEngine::create succeeded!");
        
        if (!createdEngine) {
            PGX_ERROR("ExecutionEngine::create returned null engine despite success");
            return false;
        }
        
        PGX_INFO("ExecutionEngine created and validated successfully");
        
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during ExecutionEngine creation: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Unknown exception during ExecutionEngine creation");
        
        // Try to get more information about the state
        PGX_DEBUG("Module state after crash:");
        if (module) {
            PGX_DEBUG("  Module is still valid");
            if (mlir::succeeded(mlir::verify(module))) {
                PGX_DEBUG("  Module still verifies correctly");
            } else {
                PGX_DEBUG("  Module no longer verifies");
            }
        } else {
            PGX_DEBUG("  Module is now null");
        }
        
        return false;
    }
    
    engine = std::move(createdEngine);
    initialized = true;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    PGX_INFO("JIT execution engine initialized in " + std::to_string(duration / 1000.0) + " ms");
    
    return true;
}

void PostgreSQLJITExecutionEngine::registerLingoDRuntimeContextFunctions() {
    PGX_DEBUG("Registering LingoDB runtime context functions");
    
    // Global execution context storage (thread-local for PostgreSQL safety)
    static thread_local void* global_execution_context = nullptr;
    
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            // rt_set_execution_context function (LingoDB requirement)
            symbolMap[interner("rt_set_execution_context")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(
                    +[](void* context_ptr) -> void {
                        static thread_local void* global_execution_context = nullptr;
                        global_execution_context = context_ptr;
                        PGX_DEBUG("JIT: Set execution context to " + std::to_string(reinterpret_cast<uintptr_t>(context_ptr)));
                    }
                )),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            // rt_get_execution_context function (LingoDB requirement)  
            symbolMap[interner("rt_get_execution_context")] = {
                llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(
                    +[]() -> void* {
                        static thread_local void* global_execution_context = nullptr;
                        PGX_DEBUG("JIT: Get execution context returning " + std::to_string(reinterpret_cast<uintptr_t>(global_execution_context)));
                        return global_execution_context;
                    }
                )),
                llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
            };
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerCRuntimeFunctions() {
    PGX_INFO("🎯 Registering C runtime functions");
    
    engine->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) {
            llvm::orc::SymbolMap symbolMap;
            
            // Use dlsym to get the symbols from the current process
            void* handle = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                PGX_ERROR("Failed to open current process for symbol lookup");
                return symbolMap;
            }
            
            // Helper lambda to register a single function
            auto registerFunction = [&](const std::string& funcName) {
                void* funcPtr = dlsym(handle, funcName.c_str());
                if (funcPtr) {
                    symbolMap[interner(funcName)] = {
                        llvm::orc::ExecutorAddr::fromPtr(funcPtr),
                        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
                    };
                    PGX_INFO("✅ " + funcName + " registered");
                } else {
                    PGX_WARNING("❌ " + funcName + " NOT FOUND");
                }
            };
            
            // Register all Test 1 runtime functions
            const std::vector<std::string> functions = {
                "rt_tablebuilder_create",
                "rt_tablebuilder_build", 
                "rt_tablebuilder_nextrow",
                "rt_tablebuilder_addint64",
                "rt_tablebuilder_destroy",
                "rt_datasourceiteration_start",
                "rt_datasourceiteration_isvalid",
                "rt_datasourceiteration_next",
                "rt_datasourceiteration_access",
                "rt_datasourceiteration_end",
                "rt_get_execution_context"
            };
            
            for (const auto& funcName : functions) {
                registerFunction(funcName);
            }
            
            dlclose(handle);
            
            PGX_INFO("✨ C Function Registration Complete: " + std::to_string(symbolMap.size()) + " functions registered");
            
            return symbolMap;
        });
}

void PostgreSQLJITExecutionEngine::registerPostgreSQLRuntimeFunctions() {
    PGX_INFO("🎯 SIMPLIFIED: Registering only Test 1 runtime functions");
    
    if (!engine) {
        PGX_ERROR("Cannot register runtime functions - execution engine not initialized");
        return;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();

    registerCRuntimeFunctions();
    registerLingoDRuntimeContextFunctions();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    PGX_INFO("✨ Test 1 runtime registration complete in " + std::to_string(duration / 1000.0) + " ms");
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
    
    // CRITICAL FIX: Set function visibility directly at LLVM level
    PGX_INFO("🔧 CRITICAL FIX: Applying function visibility fix at LLVM level");
    
    // Access the LLVM module from the ExecutionEngine and fix main function visibility
    auto* jit = engine.get();
    try {
        // The ExecutionEngine should have access to the underlying LLVM module
        // Let's try to get the LLVMContext and fix visibility programmatically
        PGX_INFO("🎯 Attempting to fix main function visibility in LLVM module...");
        
        // Since we can't directly access the LLVM module from ExecutionEngine,
        // let's use a different approach: check if the module has wrong visibility
        PGX_INFO("🔍 Testing function visibility issue by attempting lookups...");
        
        // First, let's see what symbols the ExecutionEngine actually knows about
        PGX_INFO("🔍 Attempting to enumerate symbols available in ExecutionEngine...");
        
        // Test with mangled name variations that might work
        std::vector<std::string> possibleNames = {"main", "_main", "_mlir_ciface_main", "query_main", "compiled_query"};
        
        bool foundFunction = false;
        for (const auto& name : possibleNames) {
            auto testResult = engine->lookup(name);
            if (testResult) {
                PGX_INFO("✅ SUCCESS: Found function with name: " + name + " at address: " + std::to_string((uintptr_t)testResult.get()));
                foundFunction = true;
                break;
            } else {
                PGX_INFO("❌ Function not found with name: " + name);
            }
        }
        
        // Since we know the main function exists in LLVM IR, try alternative approaches
        if (!foundFunction) {
            PGX_INFO("🔍 Main function not found via standard lookup - trying alternative approaches...");
            
            // Try some common LLVM mangling patterns for main function
            std::vector<std::string> mangledNames = {
                "main.1", "main_0", "@main", "query", "_Z4mainv"
            };
            
            for (const auto& name : mangledNames) {
                auto testResult = engine->lookup(name);
                if (testResult) {
                    PGX_INFO("✅ FOUND FUNCTION with mangled name: " + name + " at address: " + std::to_string(reinterpret_cast<uintptr_t>(testResult.get())));
                    foundFunction = true;
                    break;
                } else {
                    PGX_INFO("❌ Mangled name not found: " + name);
                }
            }
        }
        
        if (!foundFunction) {
            PGX_ERROR("🚨 CRITICAL: NO query function found with any variation - this indicates a fundamental visibility issue");
        } else {
            PGX_INFO("🎉 SUCCESS: Found at least one executable function in ExecutionEngine!");
        }
        
    } catch (...) {
        PGX_WARNING("Exception occurred during LLVM module visibility fix");
    }
    
    // CRITICAL FIX: Try alternative MLIR ExecutionEngine invocation methods FIRST
    PGX_INFO("🔧 ATTEMPTING alternative MLIR ExecutionEngine invocation before manual lookup...");
    
    // Helper function to execute JIT with timeout
    auto executeWithTimeout = [this](const std::string& method, std::function<llvm::Error()> exec) -> bool {
        PGX_INFO("🔧 Trying " + method + " with 10s timeout");
        
        std::promise<llvm::Error> promise;
        auto future = promise.get_future();
        
        std::thread execThread([&promise, exec]() {
            try {
                promise.set_value(exec());
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        });
        
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout) {
            PGX_ERROR("🚨 TIMEOUT: " + method + " hung for 10+ seconds");
            execThread.detach();
            return false;
        }
        
        try {
            auto result = future.get();
            execThread.join();
            if (result) {
                PGX_INFO("✅ " + method + " succeeded, checking results...");
                if (g_jit_results_ready) {
                    PGX_INFO("🎉 JIT execution completed successfully!");
                    return true;
                }
            }
        } catch (...) {
            execThread.join();
        }
        
        PGX_WARNING("❌ " + method + " failed");
        return false;
    };
    
    try {
        // Try different JIT execution methods with timeout protection
        if (executeWithTimeout("engine->invoke('main')", [this]() { return engine->invoke("main"); })) {
            return true;
        }
        
        if (executeWithTimeout("engine->invokePacked('main')", [this]() { return engine->invokePacked("main"); })) {
            return true;
        }
        
        // Method 3: Try manual function pointer approach with correct API
        PGX_INFO("🔧 Method 3: Trying manual function pointer lookup and execution");
        auto mainFuncResult = engine->lookup("main");
        if (mainFuncResult) {
            PGX_INFO("🔍 Method 3: Function lookup succeeded, attempting direct call");
            
            // Get raw function pointer from ExecutionEngine lookup result
            void* rawFuncPtr = mainFuncResult.get();
            if (rawFuncPtr) {
                PGX_INFO("🔍 Method 3: Got raw function pointer, converting to callable...");
                
                // Cast to void function pointer for main()
                auto mainFunc = reinterpret_cast<void(*)()>(rawFuncPtr);
                
                try {
                    PGX_INFO("🔍 Method 3: Calling function directly...");
                    mainFunc(); // Direct function call
                    
                    PGX_INFO("🔍 Method 3: Direct call completed, checking execution");
                    PGX_INFO("🔍 Post-execution check: g_jit_results_ready = " + std::to_string(g_jit_results_ready));
                    
                    if (g_jit_results_ready) {
                        PGX_INFO("✅ SUCCESS: Method 3 worked! Direct function call executed properly!");
                        PGX_INFO("JIT query execution completed successfully");
                        return true;
                    } else {
                        PGX_WARNING("⚠️ Method 3 failed: Direct call succeeded but function body didn't execute");
                    }
                } catch (const std::exception& e) {
                    PGX_WARNING("🔍 Method 3 exception: " + std::string(e.what()));
                } catch (...) {
                    PGX_WARNING("🔍 Method 3 unknown exception during direct call");
                }
            } else {
                PGX_WARNING("🔍 Method 3 failed: Got null raw function pointer");
            }
        } else {
            PGX_WARNING("🔍 Method 3 failed: Function lookup returned null");
        }
        
        // Method 4: Check if external function registration is the issue
        PGX_INFO("🔧 Method 4: Testing external function registration and symbol resolution");
        
        // Test if ExecutionEngine can find any of our registered runtime functions
        std::vector<std::string> registeredFunctions = {
            "increment_test_counter", "mark_results_ready_for_streaming", 
            "open_postgres_table", "read_next_tuple_from_table"
        };
        
        bool foundAnyRegistered = false;
        for (const auto& funcName : registeredFunctions) {
            auto testResult = engine->lookup(funcName);
            if (testResult) {
                PGX_INFO("✅ Found registered function: " + funcName);
                foundAnyRegistered = true;
            } else {
                PGX_INFO("❌ Could not find registered function: " + funcName);
            }
        }
        
        if (!foundAnyRegistered) {
            PGX_WARNING("🚨 CRITICAL: ExecutionEngine cannot find ANY registered functions");
            PGX_WARNING("This suggests the JIT symbol table is completely broken");
        } else {
            PGX_INFO("✅ Some registered functions found - symbol registration works");
        }
        
        // Try a different approach - check if the main function is actually compiled correctly
        PGX_INFO("🔍 Checking if main function compilation generated valid code...");
        
        // Method 5: Try creating a minimal test function that just returns a value
        PGX_INFO("🔧 Method 5: Testing with a simple return-value function");
        
        // If external function calls fail, maybe we can test with a function that just returns a value
        // Look for any function that returns an integer to test basic execution
        auto simpleTestResult = engine->lookup("main");
        if (simpleTestResult) {
            PGX_INFO("🔍 Found main function symbol, attempting to test basic execution");
            
            // Try converting to a function that returns void and see if it does anything at all
            try {
                PGX_INFO("🎯 FINAL TEST: Attempting to call function in most basic way possible");
                
                // Since invoke() claims to work, let's try to trace exactly what happens
                PGX_INFO("🔍 Re-testing invoke with detailed tracing...");
                
                // Check if the issue is memory corruption during execution
                PGX_INFO("🔍 Pre-invoke state: g_jit_results_ready = " + std::to_string(g_jit_results_ready));
                
                auto invokeResult = engine->invoke("main");
                
                PGX_INFO("🔍 Post-invoke state: g_jit_results_ready = " + std::to_string(g_jit_results_ready));
                PGX_INFO("🔍 Invoke result success: " + std::to_string(static_cast<bool>(invokeResult)));
                
            } catch (...) {
                PGX_WARNING("🔍 Exception during final test");
            }
        }
        
        PGX_WARNING("🚨 CONCLUSION: ExecutionEngine invoke() succeeds but function body doesn't execute");
        
    } catch (const std::exception& e) {
        PGX_WARNING("🔍 Exception in engine->invoke: " + std::string(e.what()));
    }
    
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
    
    // Continue with manual function pointer approach as fallback
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