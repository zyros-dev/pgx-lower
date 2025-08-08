#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

// MLIR Core Infrastructure
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"

// Dialect headers
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// AST Translation
#include "frontend/SQL/postgresql_ast_translator.h"

// Conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"

// Centralized pass pipeline
#include "mlir/Passes.h"

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

#include <fstream>

namespace mlir_runner {

// Initialize MLIR context and load dialects
// This resolves TypeID symbol linking issues by registering dialect TypeIDs
static bool initialize_mlir_context(mlir::MLIRContext& context) {
    try {
        // First validate library loading before proceeding
        if (!mlir::pgx_lower::validateLibraryLoading()) {
            PGX_ERROR("Library loading validation failed");
            return false;
        }
        
        // Register all pgx-lower passes before loading dialects
        mlir::pgx_lower::registerAllPgxLoweringPasses();
        
        // Validate pass registration
        if (!mlir::pgx_lower::validatePassRegistration()) {
            PGX_ERROR("Pass registration validation failed");
            return false;
        }
        
        // Load standard MLIR dialects
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        
        // Load custom dialects to register their TypeIDs
        context.getOrLoadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<::pgx::db::DBDialect>();
        context.getOrLoadDialect<::pgx::mlir::dsa::DSADialect>();
        
        // Validate dialect registration
        if (!mlir::pgx_lower::validateDialectRegistration()) {
            PGX_ERROR("Dialect registration validation failed");
            return false;
        }
        
        PGX_INFO("MLIR dialects and passes loaded successfully");
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("Failed to initialize MLIR context: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error during MLIR context initialization");
        return false;
    }
}

auto run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt) -> bool {
    if (!plannedStmt) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null PlannedStmt provided to MLIR runner");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    try {
        // Create MLIR context and load dialects
        mlir::MLIRContext context;
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
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        // Phase 3c: Use centralized pass pipeline for RelAlg→DB→DSA lowering
        PGX_INFO("Configuring centralized lowering pipeline");
        mlir::PassManager pm(&context);
        
        // Use centralized pipeline configuration with verification enabled
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        // Run the lowering pipeline (Phase 3a+3b+3c: RelAlg → DB → DSA)
        PGX_INFO("Running complete lowering pipeline");
        if (failed(pm.run(*module))) {
            PGX_ERROR("MLIR lowering pipeline failed (RelAlg → DB → DSA)");
            return false;
        }
        PGX_INFO("Lowering pipeline completed successfully");
        
        // Phase 4: Final verification (DSA dialect)
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Final DSA MLIR module verification failed");
            return false;
        }
        
        // TODO Phase 4b: DSA → LLVM IR lowering (next implementation phase)
        // TODO Phase 5: LLVM IR → JIT compilation (future work)
        PGX_WARNING("Phase 4b-5 not yet implemented: DSA→LLVM→JIT pipeline");
        
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
        mlir::MLIRContext context;
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
        
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Initial RelAlg MLIR module verification failed");
            return false;
        }
        
        PGX_INFO("Initial RelAlg MLIR module verified successfully");
        
        PGX_INFO("Configuring centralized lowering pipeline");
        mlir::PassManager pm(&context);
        
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        PGX_INFO("Running complete lowering pipeline");
        if (failed(pm.run(*module))) {
            PGX_ERROR("MLIR lowering pipeline failed (RelAlg → DB → DSA)");
            return false;
        }
        PGX_INFO("Lowering pipeline completed successfully");
        
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Final DB MLIR module verification failed");
            return false;
        }
        PGX_WARNING("Phase 4-5 not yet implemented: DSA→LLVM→JIT pipeline with EState");
        
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

} // namespace mlir_runner