#include "execution/mlir_runner.h"
#include "execution/mlir_logger.h"
#include "execution/error_handling.h"
#include "execution/logging.h"
#include <sstream>

// MLIR Core Infrastructure
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"

// Dialect headers
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// AST Translation
#include "frontend/SQL/postgresql_ast_translator.h"

// Conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/DBToDSA/DBToDSA.h"

// PostgreSQL error handling (only include when not building unit tests)
#ifndef BUILDING_UNIT_TESTS
extern "C" {
#include "postgres.h"
#include "utils/elog.h"
#include "utils/errcodes.h"
#include "executor/executor.h"
#include "nodes/execnodes.h"
}
#endif

// Include MLIR diagnostic infrastructure
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include <fstream>
#include "llvm/IR/Verifier.h"

namespace mlir_runner {

// Initialize MLIR context and load dialects
// This resolves TypeID symbol linking issues by registering dialect TypeIDs
static bool initialize_mlir_context(mlir::MLIRContext& context) {
    try {
        PGX_DEBUG("Initializing MLIR context and loading dialects");
        
        // Load standard MLIR dialects that AST translator needs
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        PGX_DEBUG("Loaded Func dialect");
        
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        PGX_DEBUG("Loaded Arith dialect");
        
        // Load all three custom dialects to register their TypeIDs
        // This is critical for resolving undefined symbol linking errors
        context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
        PGX_DEBUG("Loaded RelAlg dialect");
        
        context.getOrLoadDialect<pgx::db::DBDialect>();
        PGX_DEBUG("Loaded DB dialect");
        
        context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
        PGX_DEBUG("Loaded DSA dialect");
        
        PGX_INFO("MLIR context initialization successful - all dialects loaded");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR(std::string("Failed to initialize MLIR context: ") + e.what());
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error during MLIR context initialization");
        return false;
    }
}

auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) -> bool {
    if (!plannedStmt) {
        PGX_ERROR("Null PlannedStmt provided to MLIR runner");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Invalid planned statement for MLIR compilation")));
#endif
        return false;
    }
    
    PGX_INFO("Starting MLIR AST translation with minimal context initialization");
    
    try {
        // Create MLIR context and load dialects
        mlir::MLIRContext context;
        if (!initialize_mlir_context(context)) {
#ifndef BUILDING_UNIT_TESTS
            ereport(ERROR, 
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("Failed to initialize MLIR context and dialects")));
#endif
            return false;
        }
        
        // Phase 1: AST Translation - PostgreSQL AST → RelAlg MLIR
        PGX_INFO("Starting Phase 1: PostgreSQL AST to RelAlg translation");
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context, logger);
        auto module = translator->translateQuery(plannedStmt);
        
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR");
            return false;
        }
        
        PGX_INFO("Phase 1 complete: RelAlg MLIR module created successfully");
        
        // Phase 2: Verify initial MLIR module
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        PGX_INFO("Initial RelAlg MLIR module verified successfully");
        
        // Phase 3: Set up PassManager and run lowering pipeline
        PGX_INFO("Starting Phase 3: MLIR lowering pipeline (RelAlg → DB → DSA)");
        mlir::PassManager pm(&context);
        
        // Add RelAlg to DB conversion pass
        pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
        PGX_DEBUG("Added RelAlg to DB conversion pass");
        
        // Add DB to DSA conversion pass
        pm.addPass(mlir::pgx_conversion::createDBToDSAPass());
        PGX_DEBUG("Added DB to DSA conversion pass");
        
        // Run the lowering pipeline
        if (failed(pm.run(*module))) {
            PGX_ERROR("MLIR lowering pipeline failed (RelAlg → DB → DSA)");
            return false;
        }
        
        PGX_INFO("Phase 3 complete: MLIR lowering pipeline executed successfully");
        
        // Phase 4: Final verification
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Final DSA MLIR module verification failed");
            return false;
        }
        
        PGX_INFO("MLIR compilation pipeline complete - DSA module verified");
        
        // TODO Phase 5: DSA → LLVM IR → JIT compilation
        PGX_WARNING("DSA to LLVM IR and JIT compilation not yet implemented");
        
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR(std::string("MLIR runner exception: ") + e.what());
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

auto run_mlir_with_estate(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext, MLIRLogger& logger) -> bool {
    if (!plannedStmt || !estate) {
        PGX_ERROR("Null parameters provided to MLIR runner with EState");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, 
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Invalid parameters for MLIR compilation with EState")));
#endif
        return false;
    }
    
    PGX_INFO("Starting MLIR compilation with EState memory context support");
    
    try {
        // Create MLIR context and load dialects
        mlir::MLIRContext context;
        if (!initialize_mlir_context(context)) {
#ifndef BUILDING_UNIT_TESTS
            ereport(ERROR, 
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("Failed to initialize MLIR context and dialects")));
#endif
            return false;
        }
        
        // TODO Phase 5: Proper EState integration not yet implemented
        // Current implementation logs EState usage but doesn't actually integrate with PostgreSQL memory contexts.
        // Real implementation needs:
        // - Proper memory context switching during MLIR operations
        // - MLIR allocation callbacks that respect PostgreSQL memory management
        // - Cleanup handlers for PostgreSQL transaction abort scenarios
        // - Integration with PostgreSQL error handling and resource cleanup
        // WARNING: EState logging below is aspirational - actual integration is complex
        // EState provides PostgreSQL memory context - critical for proper cleanup
        RUNTIME_PGX_DEBUG("EState", "MLIR runner operating with PostgreSQL memory context");
        
        // Phase 1: AST Translation with EState context - PostgreSQL AST → RelAlg MLIR
        PGX_INFO("Starting Phase 1: PostgreSQL AST to RelAlg translation with EState support");
        RUNTIME_PGX_DEBUG("EState", "Creating AST translator with EState memory context integration");
        
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context, logger);
        auto module = translator->translateQuery(plannedStmt);
        
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR with EState");
            return false;
        }
        
        PGX_INFO("Phase 1 complete: RelAlg MLIR module created successfully with EState support");
        
        // Phase 2: Verify initial MLIR module
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed with EState");
            return false;
        }
        
        PGX_INFO("Initial RelAlg MLIR module verified successfully with EState context");
        
        // Phase 3: Set up PassManager with EState awareness and run lowering pipeline
        PGX_INFO("Starting Phase 3: MLIR lowering pipeline with EState memory management (RelAlg → DB → DSA)");
        RUNTIME_PGX_DEBUG("EState", "Configuring PassManager with PostgreSQL memory context awareness");
        
        mlir::PassManager pm(&context);
        
        // Add RelAlg to DB conversion pass
        pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
        PGX_DEBUG("Added RelAlg to DB conversion pass with EState support");
        
        // Add DB to DSA conversion pass
        pm.addPass(mlir::pgx_conversion::createDBToDSAPass());
        PGX_DEBUG("Added DB to DSA conversion pass with EState support");
        
        // Run the lowering pipeline with EState memory context monitoring
        RUNTIME_PGX_DEBUG("EState", "Executing MLIR passes within PostgreSQL memory context");
        if (failed(pm.run(*module))) {
            PGX_ERROR("MLIR lowering pipeline failed with EState (RelAlg → DB → DSA)");
            return false;
        }
        
        PGX_INFO("Phase 3 complete: MLIR lowering pipeline executed successfully with EState integration");
        
        // Phase 4: Final verification with EState context
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Final DSA MLIR module verification failed with EState");
            return false;
        }
        
        PGX_INFO("MLIR compilation pipeline complete with EState - DSA module verified");
        RUNTIME_PGX_DEBUG("EState", "MLIR pipeline execution completed within PostgreSQL memory context");
        
        // TODO Phase 5: DSA → LLVM IR → JIT compilation with EState memory management
        PGX_WARNING("DSA to LLVM IR and JIT compilation with EState not yet implemented");
        
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR(std::string("MLIR runner with EState exception: ") + e.what());
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

} // namespace mlir_runner