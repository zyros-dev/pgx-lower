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

#ifdef restrict
#define PG_RESTRICT_SAVED restrict
#undef restrict
#endif

#endif

// Include MLIR diagnostic infrastructure

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "execution/jit_execution_interface.h"
#include "mlir/Transforms/CustomPasses.h"
#include "mlir/Dialect/DB/Passes.h"

#include <mlir/InitAllPasses.h>
#include <fstream>
#include <chrono>
#include <iomanip>

// Restore PostgreSQL's restrict macro after MLIR includes
#ifndef BUILDING_UNIT_TESTS
#ifdef PG_RESTRICT_SAVED
#define restrict PG_RESTRICT_SAVED
#undef PG_RESTRICT_SAVED
#endif
#endif

extern "C" {
    struct ModuleHandle* pgx_jit_create_module_handle(void* mlir_module_ptr);
    void pgx_jit_destroy_module_handle(struct ModuleHandle* handle);
    bool test_unit_code_from_postgresql();
}

#ifndef BUILDING_UNIT_TESTS
class Phase3bMemoryGuard {
private:
    MemoryContext phase3b_context_;
    MemoryContext old_context_;
    bool active_;

public:
    Phase3bMemoryGuard() : phase3b_context_(nullptr), old_context_(nullptr), active_(false) {
        phase3b_context_ = AllocSetContextCreate(CurrentMemoryContext, "Phase3bContext", ALLOCSET_DEFAULT_SIZES);
        old_context_ = MemoryContextSwitchTo(phase3b_context_);
        active_ = true;
    }
    
    ~Phase3bMemoryGuard() {
        if (active_) {
            MemoryContextSwitchTo(old_context_);
            MemoryContextDelete(phase3b_context_);
        }
    }
    
    void deactivate() {
        if (active_) {
            MemoryContextSwitchTo(old_context_);
            MemoryContextDelete(phase3b_context_);
            active_ = false;
        }
    }
    
    Phase3bMemoryGuard(const Phase3bMemoryGuard&) = delete;
    Phase3bMemoryGuard& operator=(const Phase3bMemoryGuard&) = delete;
    Phase3bMemoryGuard(Phase3bMemoryGuard&&) = delete;
    Phase3bMemoryGuard& operator=(Phase3bMemoryGuard&&) = delete;
};
#endif

extern "C" void initialize_mlir_passes() {
    try {
       mlir::registerAllPasses();
       mlir::relalg::registerRelAlgConversionPasses();
       mlir::relalg::registerQueryOptimizationPasses();
       mlir::db::registerDBConversionPasses();
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::dsa::createLowerToStdPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::relalg::createDetachMetaDataPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::createSinkOpPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::createSimplifyMemrefsPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::createSimplifyArithmeticsPass();
       });
       ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
          return mlir::db::createSimplifyToArithPass();
       });
    } catch (const std::exception& e) {
        PGX_ERROR("Pass registration failed: " + std::string(e.what()));
    } catch (...) {
        PGX_ERROR("Pass registration failed with unknown exception");
    }
}

namespace mlir_runner {

static void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title) {
    if (!module) {
        PGX_WARNING("dumpModuleWithStats: Module is null for title: " + title);
        return;
    }
    
    auto timestamp = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    
    // Create filename with timestamp
    std::stringstream filename;
    filename << "/tmp/pgx_lower_" << title << "_" 
             << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".mlir";
    
    try {
        // Collect comprehensive module statistics
        std::map<std::string, int> dialectCounts;
        std::map<std::string, int> operationCounts;
        std::map<std::string, int> typeCounts;
        std::map<std::string, int> attributeCounts;
        int totalOperations = 0;
        int totalBlocks = 0;
        int totalRegions = 0;
        int totalValues = 0;
        
        module.walk([&](::mlir::Operation* op) {
            if (!op) return;
            
            totalOperations++;
            
            // Count by dialect
            std::string dialectName = op->getName().getDialectNamespace().str();
            if (dialectName.empty()) dialectName = "builtin";
            dialectCounts[dialectName]++;
            
            // Count by operation type
            std::string opName = op->getName().getStringRef().str();
            operationCounts[opName]++;
            
            // Count regions and blocks
            totalRegions += op->getNumRegions();
            for (auto& region : op->getRegions()) {
                totalBlocks += region.getBlocks().size();
            }
            
            // Count values (results)
            totalValues += op->getNumResults();
            
            // Count types
            for (auto result : op->getResults()) {
                std::string typeName = "unknown";
                llvm::raw_string_ostream stream(typeName);
                result.getType().print(stream);
                typeCounts[typeName]++;
            }
            
            // Count attributes
            for (auto attr : op->getAttrs()) {
                std::string attrType = attr.getName().str();
                attributeCounts[attrType]++;
            }
        });
        
        // Log comprehensive statistics
        PGX_INFO("\n\n======= " + title + " =======");
        std::stringstream timeStr;
        timeStr << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        PGX_INFO("Timestamp: " + timeStr.str());
        PGX_INFO("Output file: " + filename.str());
        
        // Overall statistics
        PGX_INFO("Module Statistics:");
        PGX_INFO("  Total Operations: " + std::to_string(totalOperations));
        PGX_INFO("  Total Blocks: " + std::to_string(totalBlocks));
        PGX_INFO("  Total Regions: " + std::to_string(totalRegions));
        PGX_INFO("  Total Values: " + std::to_string(totalValues));
        
        // Dialect breakdown
        PGX_INFO("Operations by Dialect:");
        for (const auto& [dialect, count] : dialectCounts) {
            PGX_INFO("  " + dialect + ": " + std::to_string(count));
        }
        
        // Top 10 most frequent operations
        std::vector<std::pair<int, std::string>> opsByFreq;
        for (const auto& [op, count] : operationCounts) {
            opsByFreq.emplace_back(count, op);
        }
        std::sort(opsByFreq.rbegin(), opsByFreq.rend());
        
        PGX_INFO("Top Operations by Frequency:");
        for (size_t i = 0; i < std::min(size_t(10), opsByFreq.size()); ++i) {
            PGX_INFO("  " + opsByFreq[i].second + ": " + std::to_string(opsByFreq[i].first));
        }
        
        // Type statistics (top 5)
        std::vector<std::pair<int, std::string>> typesByFreq;
        for (const auto& [type, count] : typeCounts) {
            typesByFreq.emplace_back(count, type);
        }
        std::sort(typesByFreq.rbegin(), typesByFreq.rend());
        
        PGX_INFO("Top Types by Frequency:");
        for (size_t i = 0; i < std::min(size_t(5), typesByFreq.size()); ++i) {
            PGX_INFO("  " + typesByFreq[i].second + ": " + std::to_string(typesByFreq[i].first));
        }
        
        // Module verification status
        bool isValid = ::mlir::succeeded(::mlir::verify(module));
        PGX_INFO("Module Verification: " + std::string(isValid ? "PASSED" : "FAILED"));
        
        // Print the actual MLIR code to logs as a formatted block
        try {
            std::string moduleStr;
            llvm::raw_string_ostream stream(moduleStr);
            module.print(stream);
            
            // Build formatted MLIR string with line numbers
            std::stringstream formattedMLIR;
            formattedMLIR << "\n=== MLIR MODULE CONTENT: " << title << " ===\n";
            
            std::stringstream ss(moduleStr);
            std::string line;
            int lineNum = 1;
            while (std::getline(ss, line)) {
                formattedMLIR << std::setw(3) << lineNum << ": " << line << "\n";
                lineNum++;
            }
            formattedMLIR << "=== END MLIR MODULE CONTENT ===\n";
            
            // Log the entire formatted block
            PGX_INFO(formattedMLIR.str());
            
        } catch (const std::exception& e) {
            PGX_ERROR("Failed to print MLIR module: " + std::string(e.what()));
        }
        
        // Write module to file
        std::ofstream file(filename.str());
        if (file.is_open()) {
            file << "// MLIR Module Debug Dump: " << title << "\n";
            std::stringstream genTime;
            genTime << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            file << "// Generated: " << genTime.str() << "\n";
            file << "// Total Operations: " << totalOperations << "\n";
            file << "// Module Valid: " << (isValid ? "YES" : "NO") << "\n\n";
            
            std::string moduleStr;
            llvm::raw_string_ostream stream(moduleStr);
            module.print(stream);
            file << moduleStr;
            file.close();
            
            PGX_INFO("Module dumped to: " + filename.str());
        } else {
            PGX_WARNING("Failed to open file for writing: " + filename.str());
        }
        
        PGX_INFO("=== End Module Debug Dump ===");
        PGX_INFO("\n\n");

    } catch (const std::exception& e) {
        PGX_ERROR("Exception in dumpModuleWithStats: " + std::string(e.what()));
    } catch (...) {
        PGX_ERROR("Unknown exception in dumpModuleWithStats");
    }
}

class MlirRunner {
public:
    bool executeQuery(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
        auto moduleHandle = pgx_jit_create_module_handle(&module);

        if (!moduleHandle) {
            PGX_ERROR("Failed to create module handle for JIT execution");
            auto error = pgx_jit_get_last_error();
            if (error) {
                PGX_ERROR("JIT error: " + std::string(error));
            }
            return false;
        }
        
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
        
        return true;
    }
};

static bool initialize_mlir_context(::mlir::MLIRContext& context) {
    try {
        context.disableMultithreading();
        
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::mlir::db::DBDialect>();
        context.getOrLoadDialect<::mlir::dsa::DSADialect>();
        context.getOrLoadDialect<::mlir::util::UtilDialect>();
        
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
static bool validateModuleState(mlir::ModuleOp module, const std::string& phase);
static bool runPhase3a(mlir::ModuleOp module);
static bool runPhase3b(mlir::ModuleOp module);
static bool runPhase3c(mlir::ModuleOp module);



static bool setupMLIRContextForJIT(::mlir::MLIRContext& context) {
    if (!initialize_mlir_context(context)) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Failed to initialize MLIR context and dialects");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    
    return true;
}

static bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
    auto moduleHandle = pgx_jit_create_module_handle(&module);
    if (!moduleHandle) {
        PGX_ERROR("Failed to create module handle for JIT execution");
        auto error = pgx_jit_get_last_error();
        if (error) {
            PGX_ERROR("JIT error: " + std::string(error));
        }
        return false;
    }
    
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
    
    int result = pgx_jit_execute_query(execHandle, estate, dest);
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

    return true;
}

// Run Phase 3a: RelAlg→DB+DSA+Util lowering
static bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    context.disableMultithreading();
    
    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);

    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Phase 3a: Module verification failed before lowering");
        return false;
    }

    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);

    dumpModuleWithStats(module, "MLIR before RelAlg -> Mixed");
    if (mlir::failed(pm.run(module))) {
        PGX_ERROR("Phase 3a failed: RelAlg→DB lowering error");
        return false;
    }

    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Phase 3a: Module verification failed after lowering");
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3a output")) {
        return false;
    }
    
    return true;
}

// Run Phase 3b: DB+DSA→Standard lowering
static bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
#ifndef BUILDING_UNIT_TESTS
    Phase3bMemoryGuard guard{};
#endif

    
    try {
        // Ensure all required dialects are loaded
        auto* dbDialect = context.getLoadedDialect<mlir::db::DBDialect>();
        auto* dsaDialect = context.getLoadedDialect<mlir::dsa::DSADialect>();
        auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();
        
        if (!dbDialect || !dsaDialect || !utilDialect) {
            PGX_ERROR("Phase 3b: Required dialects not loaded");
            return false;
        }
        
        if (!validateModuleState(module, "Phase 3b input")) {
            PGX_ERROR("Phase 3b: Module validation failed before running passes");
            return false;
        }
        
        if (!module) {
            PGX_ERROR("Phase 3b: Module is null!");
            return false;
        }
        
        context.disableMultithreading();
        ::mlir::PassManager pm(&context);
        pm.enableVerifier(true);
        
        if (mlir::failed(mlir::verify(module))) {
            PGX_ERROR("Phase 3b: Module verification failed before pass execution");
            return false;
        }
        
        mlir::pgx_lower::createDBDSAToStandardPipeline(pm, true);
        dumpModuleWithStats(module, "MLIR after RelAlg→DB lowering");
        
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3b failed: DB+DSA→Standard lowering error");
            return false;
        }
        
        if (!validateModuleState(module, "Phase 3b output")) {
            return false;
        }
        
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
    if (!module) {
        PGX_ERROR("Phase 3c: Module is null!");
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3c input")) {
        PGX_ERROR("Phase 3c: Invalid module state before Standard→LLVM lowering");
        return false;
    }
    
    // Add PostgreSQL-safe error handling
    volatile bool success = false;
    PG_TRY();
    {
        // Create PassManager with module context (not context pointer)
        auto* moduleContext = module.getContext();
        if (!moduleContext) {
            PGX_ERROR("Phase 3c: Module context is null!");
            success = false;
            return false;
        }
        
        ::mlir::PassManager pm(moduleContext);
        pm.enableVerifier(true);
        
        if (mlir::failed(mlir::verify(module))) {
            PGX_ERROR("Phase 3c: Module verification failed before lowering");
            success = false;
            return false;
        }
        
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        if (!module) {
            success = false;
            return false;
        }

        dumpModuleWithStats(module, "MLIR before standard -> llvm");
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3c failed: Standard→LLVM lowering error");
            success = false;
        } else {
            // Verify module after lowering
            if (mlir::failed(mlir::verify(module))) {
                PGX_ERROR("Phase 3c: Module verification failed after lowering");
                success = false;
            } else {
                success = true;
            }
        }
        dumpModuleWithStats(module, "MLIR after standard -> llvm");
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
    bool hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            // Special handling for func dialect which is allowed
            if (op->getDialect()->getNamespace() != "func") {
                hasNonLLVMOps = true;
            }
        }
    });

    if (hasNonLLVMOps) {
        PGX_ERROR("Phase 3c failed: Module contains non-LLVM operations");
        return false;
    }
    
    return true;
}

// Run complete lowering pipeline following LingoDB's unified architecture
static bool runCompleteLoweringPipeline(::mlir::ModuleOp module) {
    if (!runPhase3a(module)) {
        PGX_ERROR("Phase 3a failed");
        return false;
    }
    
    if (!runPhase3b(module)) {
        PGX_ERROR("Phase 3b failed");
        return false;
    }
    
    if (!runPhase3c(module)) {
        PGX_ERROR("Phase 3c failed");
        return false;
    }
    
    return true;
}

#ifdef POSTGRESQL_EXTENSION
auto run_mlir_with_dest_receiver(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext, DestReceiver* dest) -> bool {
    if (!plannedStmt || !estate || !dest) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null parameters provided to MLIR runner with DestReceiver");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    
    try {
        // Create and setup MLIR context
        ::mlir::MLIRContext context;
        if (!setupMLIRContextForJIT(context)) {
            return false;
        }
        
        // Phase 1: PostgreSQL AST to RelAlg translation
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context);
        if (!translator) {
            PGX_ERROR("Failed to create PostgreSQL AST translator");
            return false;
        }
        
        auto module = translator->translateQuery(plannedStmt);
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR");
            return false;
        }
        
        // Verify the generated module
        auto verifyResult = mlir::verify(*module);
        if (failed(verifyResult)) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        if (!module) {
            PGX_ERROR("Module is null after AST translation");
            return false;
        }
        
        // Phase 2-3: Run complete lowering pipeline
        if (!runCompleteLoweringPipeline(*module)) {
            return false;
        }
        
        // Phase 4: JIT execution
        if (!executeJITWithDestReceiver(*module, estate, dest)) {
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("MLIR runner exception: " + std::string(e.what()));
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("MLIR compilation failed: %s", e.what())));
#endif
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error in MLIR runner");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Unknown error during MLIR compilation")));
#endif
        return false;
    }
}
#endif

} // namespace mlir_runner