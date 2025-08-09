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

bool PostgreSQLJITExecutionEngine::validateModuleForCompilation(mlir::ModuleOp module) {
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
    auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
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

bool PostgreSQLJITExecutionEngine::compileToLLVMIR(mlir::ModuleOp module) {
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

bool PostgreSQLJITExecutionEngine::initialize(mlir::ModuleOp module) {
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
    auto moduleTranslation = [](mlir::Operation* op, llvm::LLVMContext& context) 
        -> std::unique_ptr<llvm::Module> {
        auto module = mlir::cast<mlir::ModuleOp>(op);
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

} // namespace execution
} // namespace pgx_lower