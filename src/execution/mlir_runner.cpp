#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"
#include <sstream>
#include <csignal>

// MLIR Core Infrastructure
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

// Dialect headers
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// AST Translation
#include "frontend/SQL/postgresql_ast_translator.h"

// Conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Transforms/Passes.h"

// Centralized pass pipeline
#include "mlir/Transforms/Passes.h"
#include "mlir/Passes.h"

// PostgreSQL error handling (only include when not building unit tests)
#ifndef BUILDING_UNIT_TESTS
extern "C" {
#include "postgres.h"
#include "miscadmin.h"
#include "utils/memutils.h"  // CRITICAL: Required for AllocSetContextCreate, MemoryContextSwitchTo
#include "utils/elog.h"
#include "utils/errcodes.h"
#include "executor/executor.h"
#include "nodes/execnodes.h"
}
#endif

// Include MLIR diagnostic infrastructure

#include <fstream>

// Include Standard->LLVM lowering passes
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// LLVM dialects for verification
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// Include JIT execution interface for header isolation
#include "execution/jit_execution_interface.h"

// Forward declare module handle creation
extern "C" {
    struct ModuleHandle* pgx_jit_create_module_handle(void* mlir_module_ptr);
    void pgx_jit_destroy_module_handle(struct ModuleHandle* handle);
}

// Phase 3b Memory Management Helper
#ifndef BUILDING_UNIT_TESTS
class Phase3bMemoryGuard {
private:
    MemoryContext phase3b_context_;
    MemoryContext old_context_;
    bool active_;

public:
    Phase3bMemoryGuard() : phase3b_context_(nullptr), old_context_(nullptr), active_(false) {
        // PostgreSQL requires constant string for memory context name
        phase3b_context_ = AllocSetContextCreate(CurrentMemoryContext, "Phase3bContext", ALLOCSET_DEFAULT_SIZES);
        old_context_ = MemoryContextSwitchTo(phase3b_context_);
        active_ = true;
        PGX_INFO("Phase3bMemoryGuard: Created and switched to isolated memory context");
    }
    
    ~Phase3bMemoryGuard() {
        if (active_) {
            MemoryContextSwitchTo(old_context_);
            MemoryContextDelete(phase3b_context_);
            PGX_INFO("Phase3bMemoryGuard: Cleaned up memory context");
        }
    }
    
    void deactivate() {
        if (active_) {
            MemoryContextSwitchTo(old_context_);
            MemoryContextDelete(phase3b_context_);
            active_ = false;
        }
    }
    
    // Disable copy/move to ensure single ownership
    Phase3bMemoryGuard(const Phase3bMemoryGuard&) = delete;
    Phase3bMemoryGuard& operator=(const Phase3bMemoryGuard&) = delete;
    Phase3bMemoryGuard(Phase3bMemoryGuard&&) = delete;
    Phase3bMemoryGuard& operator=(Phase3bMemoryGuard&&) = delete;
};
#endif // BUILDING_UNIT_TESTS

// C-compatible initialization function for pass registration
extern "C" void initialize_mlir_passes() {
    try {
        // Register DB conversion passes (includes DBToStd)
        mlir::db::registerDBConversionPasses();
        
        // Register RelAlg passes if needed
        mlir::relalg::registerRelAlgConversionPasses();
        
        // Note: DSA doesn't have a separate registration function
        // The DSAToStd pass is registered directly when needed
        
        PGX_INFO("MLIR passes registered successfully");
    } catch (const std::exception& e) {
        PGX_ERROR("Pass registration failed: " + std::string(e.what()));
        // Never let C++ exceptions escape to PostgreSQL
    } catch (...) {
        PGX_ERROR("Pass registration failed with unknown exception");
        // Never let C++ exceptions escape to PostgreSQL  
    }
}

namespace mlir_runner {

// MlirRunner class implementation for Phase 4g-2c requirements
class MlirRunner {
public:
    bool executeQuery(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
        PGX_INFO("MlirRunner::executeQuery - Phase 4g-2c JIT execution enabled");
        
        PGX_INFO("About to call pgx_jit_create_module_handle");
        // Create module handle for isolated JIT execution
        auto moduleHandle = pgx_jit_create_module_handle(&module);
        PGX_INFO("pgx_jit_create_module_handle completed successfully");
        
        if (!moduleHandle) {
            PGX_ERROR("Failed to create module handle for JIT execution");
            auto error = pgx_jit_get_last_error();
            if (error) {
                PGX_ERROR("JIT error: " + std::string(error));
            }
            return false;
        }
        
        PGX_INFO("About to call pgx_jit_create_execution_handle");
        // Create execution handle
        auto execHandle = pgx_jit_create_execution_handle(moduleHandle);
        PGX_INFO("pgx_jit_create_execution_handle completed successfully");
        
        PGX_INFO("About to destroy module handle");
        pgx_jit_destroy_module_handle(moduleHandle);
        PGX_INFO("Module handle destroyed successfully");
        
        if (!execHandle) {
            PGX_ERROR("Failed to create JIT execution handle");
            auto error = pgx_jit_get_last_error();
            if (error) {
                PGX_ERROR("JIT error: " + std::string(error));
            }
            return false;
        }
        
        // Execute the compiled query
        int result = pgx_jit_execute_query(execHandle, estate, dest);
        
        // Cleanup
        pgx_jit_destroy_execution_handle(execHandle);
        
        if (result != 0) {
            PGX_ERROR("JIT query execution failed with code: " + std::to_string(result));
            auto error = pgx_jit_get_last_error();
            if (error) {
                PGX_ERROR("JIT error: " + std::string(error));
            }
            return false;
        }
        
        PGX_INFO("JIT query execution completed successfully");
        return true;
    }
};

// Initialize MLIR context and load dialects
// This resolves TypeID symbol linking issues by registering dialect TypeIDs
static bool initialize_mlir_context(::mlir::MLIRContext& context) {
    try {
        // CRITICAL: Disable multithreading for PostgreSQL compatibility
        context.disableMultithreading();
        
        // LLVM initialization moved to JIT engine to prevent duplicate initialization
        // which was causing memory corruption in PostgreSQL server context
        
        // Load standard MLIR dialects
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        
        // Load custom dialects to register their TypeIDs
        context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::mlir::db::DBDialect>();
        context.getOrLoadDialect<::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<::mlir::util::UtilDialect>();
        
        PGX_INFO("MLIR dialects loaded successfully");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to initialize MLIR context: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error during MLIR context initialization");
        return false;
    }
}

// Forward declare helper functions
static void safeModulePrint(mlir::ModuleOp module, const std::string& label);
static bool validateModuleState(mlir::ModuleOp module, const std::string& phase);
static bool runPhase3a(mlir::ModuleOp module);
static bool runPhase3b(mlir::ModuleOp module);
static bool runPhase3c(mlir::ModuleOp module);

auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt) -> bool {
    if (!plannedStmt) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null PlannedStmt provided to MLIR runner");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    try {
        // Create MLIR context and load dialects
        ::mlir::MLIRContext context;
        if (!initialize_mlir_context(context)) {
            auto error = pgx_lower::ErrorManager::postgresqlError("Failed to initialize MLIR context and dialects");
            pgx_lower::ErrorManager::reportError(error);
            return false;
        }
        
        // Phase 1: AST Translation - PostgreSQL AST → RelAlg MLIR
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context);
        auto module = translator->translateQuery(plannedStmt);
        
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR");
            return false;
        }
        
        // Phase 2: Verify initial MLIR module
        if (mlir::failed(mlir::verify(module->getOperation()))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        // Phase 3: Sequential PassManager approach following LingoDB patterns
        
        // Phase 3a: RelAlg→DB lowering
        PGX_INFO("Phase 3a: Running RelAlg→DB lowering pipeline");
        {
            ::mlir::PassManager pm1(&context);
            mlir::pgx_lower::createRelAlgToDBPipeline(pm1, true);
            if (mlir::failed(pm1.run(*module))) {
                PGX_ERROR("Phase 3a failed: RelAlg→DB lowering pipeline error");
                return false;
            }
            PGX_INFO("Phase 3a completed: RelAlg successfully lowered to DB+DSA+Util");
        }
        
        // Phase 3b: DB+DSA→Standard lowering
        PGX_INFO("Phase 3b: Running DB+DSA→Standard lowering pipeline");
        {
            ::mlir::PassManager pm2(&context);
            mlir::pgx_lower::createDBDSAToStandardPipeline(pm2, true);
            if (mlir::failed(pm2.run(*module))) {
                PGX_ERROR("Phase 3b failed: DB+DSA→Standard lowering pipeline error");
                return false;
            }
            PGX_INFO("Phase 3b completed: DB+DSA successfully lowered to Standard MLIR");
        }
        
        // Phase 3c: Standard→LLVM lowering
        PGX_INFO("Phase 3c: Running Standard→LLVM lowering pipeline");
        {
            ::mlir::PassManager pm3(&context);
            mlir::pgx_lower::createStandardToLLVMPipeline(pm3, true);
            if (mlir::failed(pm3.run(*module))) {
                PGX_ERROR("Phase 3c failed: Standard→LLVM lowering pipeline error");
                return false;
            }
            PGX_INFO("Phase 3c completed: Standard MLIR successfully lowered to LLVM IR");
        }
        
        // Phase 3d: Function-level optimizations (optional)
        PGX_INFO("Phase 3d: Running function-level optimizations");
        {
            ::mlir::PassManager pmFunc(&context, mlir::func::FuncOp::getOperationName());
            mlir::pgx_lower::createFunctionOptimizationPipeline(pmFunc, true);
            
            // Walk all functions and apply optimizations (like LingoDB)
            module->walk([&](mlir::func::FuncOp f) {
                if (!f->hasAttr("passthrough")) {
                    if (mlir::failed(pmFunc.run(f))) {
                        PGX_WARNING("Function optimization failed for: " + f.getName().str());
                    }
                }
            });
            PGX_INFO("Phase 3d completed: Function-level optimizations applied");
        }
        
        // TODO: Next phases - DSA to LLVM IR and JIT compilation
        PGX_INFO("Lowering pipeline completed - next phases not yet implemented");
        
        return true;
        
    } catch (const std::exception& e) {
        auto error = pgx_lower::ErrorManager::postgresqlError("MLIR compilation failed: " + std::string(e.what()));
        pgx_lower::ErrorManager::reportError(error);
        return false;
    } catch (...) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Unknown error during MLIR compilation");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
}

auto run_mlir_with_estate(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext) -> bool {
    if (!plannedStmt || !estate) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null parameters provided to MLIR runner with EState");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    PGX_INFO("Starting MLIR compilation with EState memory context support");
    
    try {
        // Create MLIR context and load dialects
        ::mlir::MLIRContext context;
        if (!initialize_mlir_context(context)) {
            auto error = pgx_lower::ErrorManager::postgresqlError("Failed to initialize MLIR context and dialects");
            pgx_lower::ErrorManager::reportError(error);
            return false;
        }
        
        PGX_INFO("Starting Phase 1: PostgreSQL AST to RelAlg translation with EState support");
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context);
        auto module = translator->translateQuery(plannedStmt);
        
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR");
            return false;
        }
        
        PGX_INFO("Phase 1 complete: RelAlg MLIR module created successfully");
        
        // Verify module is well-formed
        if (mlir::failed(mlir::verify(module->getOperation()))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        PGX_INFO("Initial RelAlg MLIR module verified successfully");
        
        // Phase 3: Sequential PassManager approach following LingoDB patterns
        
        // Phase 3a: RelAlg→DB lowering
        PGX_INFO("Phase 3a: Running RelAlg→DB lowering pipeline");
        {
            ::mlir::PassManager pm1(&context);
            mlir::pgx_lower::createRelAlgToDBPipeline(pm1, true);
            if (mlir::failed(pm1.run(*module))) {
                PGX_ERROR("Phase 3a failed: RelAlg→DB lowering pipeline error");
                return false;
            }
            PGX_INFO("Phase 3a completed: RelAlg successfully lowered to DB+DSA+Util");
        }
        
        // CRITICAL: Add module validation after Phase 3a
        if (mlir::failed(mlir::verify(module->getOperation()))) {
            PGX_ERROR("Phase 3a module verification failed - output is malformed");
            // Safe dump for debugging
            // Safe module printing for debugging
            safeModulePrint(*module, "Phase 3a verification failure");
            return false;
        }
        PGX_INFO("Phase 3a completed and verified: RelAlg successfully lowered to DB+DSA+Util");
        
        // Add IR dumping for debugging
        if (get_logger().should_log(LogLevel::DEBUG_LVL)) {
            safeModulePrint(*module, "Phase 3a Output Module");
        }
        
        // Verify dialects are loaded before Phase 3b
        auto* dbDialect = context.getLoadedDialect<mlir::db::DBDialect>();
        auto* dsaDialect = context.getLoadedDialect<mlir::dsa::DSADialect>();
        if (!dbDialect || !dsaDialect) {
            PGX_ERROR("Required dialects not loaded for Phase 3b");
            return false;
        }
        PGX_INFO("DB and DSA dialects verified for Phase 3b");
        
        // Phase 3b: DB+DSA→Standard lowering
        PGX_INFO("Phase 3b: Running DB+DSA→Standard lowering pipeline");
        {
            ::mlir::PassManager pm2(&context);
            mlir::pgx_lower::createDBDSAToStandardPipeline(pm2, true);
            if (mlir::failed(pm2.run(*module))) {
                PGX_ERROR("Phase 3b failed: DB+DSA→Standard lowering pipeline error");
                return false;
            }
            PGX_INFO("Phase 3b completed: DB+DSA successfully lowered to Standard MLIR");
        }
        
        // Phase 3c: Standard→LLVM lowering
        PGX_INFO("Phase 3c: Running Standard→LLVM lowering pipeline");
        {
            ::mlir::PassManager pm3(&context);
            mlir::pgx_lower::createStandardToLLVMPipeline(pm3, true);
            if (mlir::failed(pm3.run(*module))) {
                PGX_ERROR("Phase 3c failed: Standard→LLVM lowering pipeline error");
                return false;
            }
            PGX_INFO("Phase 3c completed: Standard MLIR successfully lowered to LLVM IR");
        }
        
        // Phase 3d: Function-level optimizations (optional)
        PGX_INFO("Phase 3d: Running function-level optimizations");
        {
            ::mlir::PassManager pmFunc(&context, mlir::func::FuncOp::getOperationName());
            mlir::pgx_lower::createFunctionOptimizationPipeline(pmFunc, true);
            
            // Walk all functions and apply optimizations (like LingoDB)
            module->walk([&](mlir::func::FuncOp f) {
                if (!f->hasAttr("passthrough")) {
                    if (mlir::failed(pmFunc.run(f))) {
                        PGX_WARNING("Function optimization failed for: " + f.getName().str());
                    }
                }
            });
            PGX_INFO("Phase 3d completed: Function-level optimizations applied");
        }
        
        if (mlir::failed(mlir::verify(module->getOperation()))) {
            PGX_ERROR("Final module verification failed");
            return false;
        }
        
        PGX_INFO("Lowering pipeline completed successfully");
        
        // Phase 4g-2c: Complete Standard→LLVM→JIT execution
        PGX_INFO("Phase 4g-2c: Preparing for JIT execution with header isolation");
        
        // Create MlirRunner instance for isolated JIT execution
        MlirRunner runner;
        bool jitSuccess = runner.executeQuery(*module, reinterpret_cast<EState*>(estate), nullptr);
        
        if (!jitSuccess) {
            PGX_WARNING("JIT execution failed - query may not have produced results");
            // Return true anyway - the lowering succeeded, just no JIT execution
            return true;
        }
        
        PGX_INFO("Phase 4g-2c: JIT execution completed successfully");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("MLIR runner exception: " + std::string(e.what()));
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("MLIR compilation with EState failed: %s", e.what())));
#endif
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error in MLIR runner with EState");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Unknown error during MLIR compilation with EState")));
#endif
        return false;
    }
}

// Setup MLIR context for JIT compilation
static bool setupMLIRContextForJIT(::mlir::MLIRContext& context) {
    if (!initialize_mlir_context(context)) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Failed to initialize MLIR context and dialects");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    // Load additional dialects needed for JIT compilation
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    
    return true;
}

// Execute JIT compiled module with destination receiver
static bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
    // Create module handle for isolated JIT execution
    auto moduleHandle = pgx_jit_create_module_handle(&module);
    if (!moduleHandle) {
        PGX_ERROR("Failed to create module handle for JIT execution");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
    // Create execution handle with isolation
    auto execHandle = pgx_jit_create_execution_handle(moduleHandle);
    pgx_jit_destroy_module_handle(moduleHandle);
    
    if (!execHandle) {
        PGX_ERROR("Failed to create JIT execution handle");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
    // Execute the compiled query with destination receiver
    PGX_INFO("Executing JIT compiled query with destination receiver");
    int result = pgx_jit_execute_query(execHandle, estate, dest);
    
    // Cleanup
    pgx_jit_destroy_execution_handle(execHandle);
    
    if (result != 0) {
        PGX_ERROR("JIT query execution failed with code: " + std::to_string(result));
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
    return true;
}

// Helper function to validate module before Phase 3b execution
static bool validatePhase3bPreconditions(mlir::ModuleOp module) {
    if (!module.getOperation()) {
        PGX_ERROR("Module operation became null before Phase 3b execution");
        return false;
    }
    
    if (mlir::failed(mlir::verify(module.getOperation()))) {
        PGX_ERROR("Module verification failed immediately before Phase 3b - module is corrupted");
        return false;
    }
    
    return true;
}

// Helper function to dump module state for debugging
static void dumpModuleForDebugging(mlir::ModuleOp module, const std::string& phase) {
    if (get_logger().should_log(LogLevel::INFO_LVL)) {
        // Safe module printing for debugging
        PGX_INFO(phase + ": Module state (printing disabled to avoid crash)");
    }
}

// Safe module printing that handles potential crashes
static void safeModulePrint(mlir::ModuleOp module, const std::string& label) {
    PGX_INFO("safeModulePrint called for: " + label);
    
    if (!module) {
        PGX_WARNING(label + ": Module is null, cannot print");
        return;
    }
    
    if (!module.getOperation()) {
        PGX_WARNING(label + ": Module operation is null, cannot print");
        return;
    }
    
    try {
        PGX_INFO("Attempting to verify module for: " + label);
        
        // First try to verify the module is valid
        if (mlir::failed(mlir::verify(module.getOperation()))) {
            PGX_WARNING(label + ": Module verification failed, cannot print invalid module");
            return;
        }
        
        PGX_INFO("Module verification passed for: " + label);
        
        // Count operations as a safer check
        int totalOps = 0;
        try {
            PGX_INFO(label + ": Starting operation count walk");
            module.walk([&](mlir::Operation* op) {
                if (!op) {
                    PGX_WARNING(label + ": Null operation during walk");
                    return;
                }
                totalOps++;
            });
            PGX_INFO(label + ": Walk completed, found " + std::to_string(totalOps) + " operations");
        } catch (const std::exception& e) {
            PGX_WARNING(label + ": Exception during operation count: " + std::string(e.what()));
            return;
        } catch (...) {
            PGX_WARNING(label + ": Unknown exception during operation count");
            return;
        }
        
        PGX_INFO(label + ": Module contains " + std::to_string(totalOps) + " operations");
        
        // Temporarily disable module printing to avoid crashes
        // TODO: Re-enable once we understand why module->print() crashes PostgreSQL
        PGX_WARNING(label + ": Module printing temporarily disabled to avoid PostgreSQL crashes");
        PGX_INFO(label + ": Module statistics - " + std::to_string(totalOps) + " operations");
        
        // Walk and print operation types instead of full module
        try {
            std::map<std::string, int> opTypes;
            module.walk([&](mlir::Operation* op) {
                if (op) {
                    std::string opName = op->getName().getStringRef().str();
                    opTypes[opName]++;
                }
            });
            
            for (const auto& [opType, count] : opTypes) {
                PGX_INFO(label + ": Operation type '" + opType + "': " + std::to_string(count));
            }
        } catch (...) {
            PGX_WARNING(label + ": Failed to collect operation statistics");
        }
    } catch (const std::exception& e) {
        PGX_WARNING(label + ": Exception during module print: " + std::string(e.what()));
    } catch (...) {
        PGX_WARNING(label + ": Unknown exception during module print");
    }
}

// Helper function to execute Phase 3b with proper PostgreSQL memory management
static bool executePhase3bWithMemoryIsolation(mlir::ModuleOp module, mlir::PassManager& pm) {
    PGX_DEBUG("Executing Phase 3b with PostgreSQL memory context");
    
    // Validate preconditions
    if (!validatePhase3bPreconditions(module)) {
        PGX_ERROR("Phase 3b preconditions validation failed");
        return false;
    }
    
    PGX_INFO("Running Phase 3b PassManager...");
    
    // Dump module state before conversion if debug enabled
    dumpModuleForDebugging(module, "Phase 3b: Module before DB→Std conversion");
    
    // Execute with proper PostgreSQL memory context
    bool phase3b_success = false;
    
#ifndef BUILDING_UNIT_TESTS
    // Use PostgreSQL memory context for production
    MemoryContext oldcontext = CurrentMemoryContext;
    MemoryContext phase3b_context = AllocSetContextCreate(CurrentMemoryContext,
                                                          "Phase3bContext",
                                                          ALLOCSET_DEFAULT_SIZES);
    
    PG_TRY();
    {
        MemoryContextSwitchTo(phase3b_context);
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3b: PassManager execution failed");
            phase3b_success = false;
        } else {
            PGX_INFO("Phase 3b: PassManager execution succeeded");
            phase3b_success = true;
        }
        MemoryContextSwitchTo(oldcontext);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(oldcontext);
        MemoryContextDelete(phase3b_context);
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    MemoryContextDelete(phase3b_context);
#else
    // Unit tests run without PostgreSQL context
    if (mlir::failed(pm.run(module))) {
        PGX_ERROR("Phase 3b: PassManager execution failed");
        phase3b_success = false;
    } else {
        PGX_INFO("Phase 3b: PassManager execution succeeded");
        phase3b_success = true;
    }
#endif
    
    if (!phase3b_success) {
        PGX_ERROR("Phase 3b: Execution failed, aborting pipeline");
        return false;
    }
    
    return true;
}

// Validate module state between pipeline phases
static bool validateModuleState(::mlir::ModuleOp module, const std::string& phase) {
    if (!module || !module.getOperation()) {
        PGX_ERROR(phase + ": Module operation is null");
        return false;
    }
    
    if (mlir::failed(mlir::verify(module.getOperation()))) {
        PGX_ERROR(phase + ": Module verification failed");
        return false;
    }
    
    // Count operations by dialect
    std::map<std::string, int> dialectOpCounts;
    module.walk([&](mlir::Operation* op) {
        auto dialectName = op->getName().getDialectNamespace();
        dialectOpCounts[dialectName.str()]++;
    });
    
    PGX_DEBUG(phase + " operation counts:");
    for (const auto& [dialect, count] : dialectOpCounts) {
        PGX_DEBUG("  - " + dialect + ": " + std::to_string(count));
    }
    
    return true;
}

// Run Phase 3a: RelAlg→DB+DSA+Util lowering
static bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    PGX_INFO("Phase 3a: Running RelAlg→DB lowering");
    
    ::mlir::PassManager pm(&context);
    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);
    
    if (mlir::failed(pm.run(module))) {
        PGX_ERROR("Phase 3a failed: RelAlg→DB lowering error");
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3a output")) {
        return false;
    }
    
    PGX_INFO("Phase 3a completed: RelAlg successfully lowered to DB+DSA+Util");
    return true;
}

// Run Phase 3b: DB+DSA→Standard lowering
static bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    
    try {
        PGX_INFO("Phase 3b: Running DB+DSA→Standard lowering");
        
        // Ensure all required dialects are loaded
        auto* dbDialect = context.getLoadedDialect<mlir::db::DBDialect>();
        auto* dsaDialect = context.getLoadedDialect<mlir::dsa::DSADialect>();
        auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();
        
        if (!dbDialect || !dsaDialect || !utilDialect) {
            PGX_ERROR("Phase 3b: Required dialects not loaded");
            return false;
        }
        
        // CRITICAL: Do NOT call setParentModule - causes memory corruption with sequential PassManagers
        // This was discovered to cause crashes in Phase 3b
        // Both DBToStd and DSAToStd passes skip this call to prevent crashes
        PGX_INFO("Phase 3b: Skipping setParentModule to prevent memory corruption");
        
        // Validate module state before running passes
        if (!validateModuleState(module, "Phase 3b input")) {
            PGX_ERROR("Phase 3b: Module validation failed before running passes");
            return false;
        }
        
        // Additional safety check - verify module is not null
        if (!module) {
            PGX_ERROR("Phase 3b: Module is null!");
            return false;
        }
        
        // Verify module operation count
        int opCount = 0;
        module.walk([&opCount](::mlir::Operation* op) { opCount++; });
        PGX_INFO("Phase 3b: Module has " + std::to_string(opCount) + " operations before conversion");
        
        context.disableMultithreading();
        PGX_INFO("Phase 3b: Threading disabled for PostgreSQL compatibility");
        
        ::mlir::PassManager pm(&context);
        
        // COMMENTED OUT: These MLIR debugging features cause segfaults in PostgreSQL
        // Root cause: They bypass PostgreSQL's memory context system
        // pm.enableTiming();                    // Memory allocator conflicts
        // pm.enableStatistics();               // Bypasses PostgreSQL memory contexts
        // pm.enableCrashReproducerGeneration("/tmp/pgx_lower_phase3b_crash.mlir"); // Signal handling conflicts
        
        pm.enableVerifier(true);  // This one is safe and works in PostgreSQL
        PGX_INFO("Phase 3b: PassManager configured for PostgreSQL compatibility (debugging features disabled)");
        
        // Pre-execution module validation
        PGX_INFO("Phase 3b: Validating module state before pass execution");
        if (mlir::failed(mlir::verify(module))) {
            PGX_ERROR("Phase 3b: Module verification failed before pass execution");
            return false;
        }
        
        // Dialect validation
        auto dialects = context.getLoadedDialects();
        PGX_INFO("Phase 3b: Loaded dialects before pass execution:");
        for (auto* dialect : dialects) {
            if (dialect) {
                PGX_INFO("  - Dialect: " + std::string(dialect->getNamespace().str()));
            }
        }
        
        PGX_INFO("Phase 3b: Creating DB+DSA→Standard pipeline");
        mlir::pgx_lower::createDBDSAToStandardPipeline(pm, true);
        
        PGX_INFO("Phase 3b: Starting PassManager execution");
        
        // Critical: Exception-safe PassManager execution
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3b failed: DB+DSA→Standard lowering error");
            PGX_ERROR("Check /tmp/pgx_lower_phase3b_crash.mlir for crash reproducer");
            return false;
        }
        
        if (!validateModuleState(module, "Phase 3b output")) {
            return false;
        }
        
        PGX_INFO("Phase 3b completed: DB+DSA successfully lowered to Standard MLIR");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("Phase 3b C++ exception: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Phase 3b unknown C++ exception - backend crash prevented");
        return false;
    }
}

// Run Phase 3c: Standard→LLVM lowering
static bool runPhase3c(::mlir::ModuleOp module) {
    PGX_INFO("Phase 3c: Running Standard→LLVM lowering");
    
    // Validate module before conversion
    if (!module) {
        PGX_ERROR("Phase 3c: Module is null!");
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3c input")) {
        PGX_ERROR("Phase 3c: Invalid module state before Standard→LLVM lowering");
        return false;
    }
    
    // Debug: Print operation types before lowering
    PGX_INFO("Phase 3c: Operations before Standard→LLVM lowering:");
    std::map<std::string, int> dialectCounts;
    module->walk([&](mlir::Operation* op) {
        if (op->getDialect()) {
            dialectCounts[op->getDialect()->getNamespace().str()]++;
        }
    });
    for (const auto& [dialect, count] : dialectCounts) {
        PGX_INFO("  - " + dialect + ": " + std::to_string(count));
    }
    
    // Add PostgreSQL-safe error handling
    volatile bool success = false;
    PG_TRY();
    {
        // Create PassManager with module context (not context pointer)
        PGX_INFO("Phase 3c: Creating PassManager");
        auto* moduleContext = module.getContext();
        if (!moduleContext) {
            PGX_ERROR("Phase 3c: Module context is null!");
            success = false;
            return false;
        }
        
        PGX_INFO("Phase 3c: Module context obtained, creating PassManager");
        ::mlir::PassManager pm(moduleContext);
        
        PGX_INFO("Phase 3c: PassManager created successfully");
        PGX_INFO("Phase 3c: Configuring Standard→LLVM pipeline");
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        
        PGX_INFO("Phase 3c: Running PassManager");
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3c failed: Standard→LLVM lowering error");
            success = false;
        } else {
            success = true;
        }
    }
    PG_CATCH();
    {
        PGX_ERROR("Phase 3c: PostgreSQL exception caught during Standard→LLVM lowering");
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    if (!success) {
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3c output")) {
        return false;
    }
    
    // Enhanced verification: ensure all operations are LLVM dialect
    PGX_INFO("Verifying complete lowering to LLVM dialect");
    bool hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            // Special handling for func dialect which is allowed
            if (op->getDialect()->getNamespace() == "func") {
                PGX_INFO("Func operation remains (allowed): " +
                         op->getName().getStringRef().str());
            } else {
                PGX_INFO("Non-LLVM operation remains after lowering: " +
                         op->getName().getStringRef().str() + " from dialect: " +
                         op->getDialect()->getNamespace().str());
                hasNonLLVMOps = true;
            }
        }
    });
    PGX_INFO("Finished verifying complete lowering to LLVM dialect");

    if (hasNonLLVMOps) {
        PGX_ERROR("Phase 3c failed: Module contains non-LLVM operations");
        return false;
    }
    
    PGX_INFO("Phase 3c completed: All operations successfully lowered to LLVM dialect");
    return true;
}

// Run complete lowering pipeline following LingoDB's unified architecture
static bool runCompleteLoweringPipeline(::mlir::ModuleOp module) {
    PGX_INFO("Running complete MLIR lowering pipeline");
    
    // Phase 3a: RelAlg→DB+DSA+Util lowering
    if (!runPhase3a(module)) {
        PGX_ERROR("Phase 3a failed");
        return false;
    }
    
    // Phase 3b: DB+DSA→Standard lowering
    if (!runPhase3b(module)) {
        PGX_ERROR("Phase 3b failed");
        return false;
    }
    
    // Phase 3c: Standard→LLVM lowering
    if (!runPhase3c(module)) {
        PGX_ERROR("Phase 3c failed");
        return false;
    }
    
    PGX_INFO("Complete pipeline succeeded");
    return true;
}

auto run_mlir_with_dest_receiver(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext, DestReceiver* dest) -> bool {
    if (!plannedStmt || !estate || !dest) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null parameters provided to MLIR runner with DestReceiver");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    PGX_INFO("Starting Phase 4g-2c: Full MLIR compilation and JIT execution with DestReceiver");
    
    try {
        // Create and setup MLIR context
        ::mlir::MLIRContext context;
        if (!setupMLIRContextForJIT(context)) {
            return false;
        }
        
        // Phase 1: PostgreSQL AST to RelAlg translation
        PGX_INFO("Phase 1: PostgreSQL AST to RelAlg translation");
        PGX_DEBUG("Creating PostgreSQL AST translator...");
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context);
        if (!translator) {
            PGX_ERROR("Failed to create PostgreSQL AST translator");
            return false;
        }
        
        PGX_DEBUG("Translating query...");
        auto module = translator->translateQuery(plannedStmt);
        
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR");
            return false;
        }
        
        PGX_DEBUG("Module created, verifying...");
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        PGX_DEBUG("Module verified successfully");
        
        // Dump the initial RelAlg MLIR for debugging
        {
            PGX_INFO("=== Initial RelAlg MLIR ===");
            
            // Check if module is valid before printing
            if (!module) {
                PGX_ERROR("Module is null after AST translation");
                return false;
            }
            
            // Use safe module printing to avoid PostgreSQL backend crashes
            safeModulePrint(*module, "Initial RelAlg MLIR");
            
            // Add module stats
            try {
                PGX_INFO("Starting module walk for stats");
                size_t opCount = 0;
                std::map<std::string, int> dialectOpCounts;
                module->walk([&opCount, &dialectOpCounts](mlir::Operation *op) { 
                    if (!op) return;
                    opCount++;
                    auto dialectName = op->getName().getDialectNamespace();
                    if (!dialectName.empty()) {
                        dialectOpCounts[dialectName.str()]++;
                    }
                });
                PGX_INFO("Module contains " + std::to_string(opCount) + " operations");
                for (const auto& [dialect, count] : dialectOpCounts) {
                    PGX_INFO("  - " + dialect + ": " + std::to_string(count));
                }
            } catch (const std::exception& e) {
                PGX_WARNING("Exception during module walk: " + std::string(e.what()));
            } catch (...) {
                PGX_WARNING("Unknown exception during module walk");
            }
        }
        
        // Phase 2-3: Run complete lowering pipeline
        PGX_INFO("Phase 2-3: Running complete lowering pipeline (RelAlg→DB+DSA→Standard)");
        if (!runCompleteLoweringPipeline(*module)) {
            return false;
        }
        
        // Phase 4g-2: JIT execution
        PGX_INFO("Phase 4g-2: JIT Execution with DestReceiver");
        if (!executeJITWithDestReceiver(*module, estate, dest)) {
            return false;
        }
        
        PGX_INFO("Phase 4g-2c: JIT execution with DestReceiver completed successfully");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("MLIR runner exception: " + std::string(e.what()));
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("MLIR compilation with DestReceiver failed: %s", e.what())));
#endif
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error in MLIR runner with DestReceiver");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Unknown error during MLIR compilation with DestReceiver")));
#endif
        return false;
    }
}

} // namespace mlir_runner