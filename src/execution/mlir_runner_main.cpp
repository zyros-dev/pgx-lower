#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

// MLIR verification includes
#include "mlir/IR/Verifier.h"

// AST Translation
#include "frontend/SQL/postgresql_ast_translator.h"

// PostgreSQL includes for main interface
#ifndef BUILDING_UNIT_TESTS
extern "C" {
#include "postgres.h"
#include "utils/errcodes.h"
#include "utils/elog.h"
#include "executor/executor.h"
#include "nodes/execnodes.h"
}
#endif

namespace mlir_runner {

// Main entry point for PostgreSQL extension - orchestrates the complete pipeline
#ifdef POSTGRESQL_EXTENSION
auto run_mlir_with_dest_receiver(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext, void* dest) -> bool {
    if (!plannedStmt || !estate || !dest) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null parameters provided to MLIR runner with DestReceiver");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }
    
    try {
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
        auto verifyResult = ::mlir::verify(*module);
        if (::mlir::failed(verifyResult)) {
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