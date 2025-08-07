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
        // Load standard MLIR dialects
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        
        // Load custom dialects to register their TypeIDs
        context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<pgx::db::DBDialect>();
        context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
        
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
        auto translator = postgresql_ast::createPostgreSQLASTTranslator(context, logger);
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
        
        // Phase 3: Set up PassManager and run lowering pipeline
        mlir::PassManager pm(&context);
        pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
        pm.addPass(mlir::pgx_conversion::createDBToDSAPass());
        
        // Run the lowering pipeline
        if (failed(pm.run(*module))) {
            PGX_ERROR("MLIR lowering pipeline failed (RelAlg → DB → DSA)");
            return false;
        }
        
        // Phase 4: Final verification
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Final DSA MLIR module verification failed");
            return false;
        }
        
        // TODO Phase 5: DSA → LLVM IR → JIT compilation
        PGX_WARNING("DSA to LLVM IR and JIT compilation not yet implemented");
        
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
        
        // Phase 1: AST Translation - PostgreSQL AST → RelAlg MLIR
        PGX_INFO("Starting Phase 1: PostgreSQL AST to RelAlg translation with EState support");
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
        
        // Phase 3: Set up PassManager and run lowering pipeline
        mlir::PassManager pm(&context);
        pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
        pm.addPass(mlir::pgx_conversion::createDBToDSAPass());
        
        // Run the lowering pipeline
        if (failed(pm.run(*module))) {
            PGX_ERROR("MLIR lowering pipeline failed (RelAlg → DB → DSA)");
            return false;
        }
        
        // Phase 4: Final verification
        if (failed(mlir::verify(*module))) {
            PGX_ERROR("Final DSA MLIR module verification failed");
            return false;
        }
        
        // TODO Phase 5: DSA → LLVM IR → JIT compilation not yet implemented
        PGX_WARNING("Full MLIR compilation with EState pipeline not yet implemented");
        
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