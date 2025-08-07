#include "execution/mlir_runner.h"
#include "execution/mlir_logger.h"
#include "execution/error_handling.h"
#include "execution/logging.h"
#include <sstream>

// MLIR Core Infrastructure
#include "mlir/IR/MLIRContext.h"

// Dialect headers
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"

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
        
        // Load all three dialects to register their TypeIDs
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
        
        // For now, just verify dialects are loaded and return success
        // Full AST translation pipeline will be implemented incrementally
        PGX_INFO("MLIR context successfully initialized - dialects loaded and TypeIDs registered");
        
        // TODO Phase 4: Implement PostgreSQL AST → RelAlg → DB → DSA → LLVM IR → JIT pipeline
        PGX_WARNING("Full MLIR compilation pipeline not yet implemented - returning early success");
        
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
        
        // EState provides PostgreSQL memory context - critical for proper cleanup
        RUNTIME_PGX_DEBUG("EState", "MLIR runner operating with PostgreSQL memory context");
        
        // For now, just verify dialects are loaded and return success
        // Full EState integration will be implemented incrementally
        PGX_INFO("MLIR context with EState successfully initialized - dialects loaded and TypeIDs registered");
        
        // TODO Phase 5: Implement full pipeline with proper EState memory management
        PGX_WARNING("Full MLIR compilation with EState pipeline not yet implemented - returning early success");
        
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