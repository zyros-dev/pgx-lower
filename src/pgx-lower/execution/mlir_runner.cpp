#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

// AST Translation
#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"


#ifndef BUILDING_UNIT_TESTS
extern "C" {
#include "postgres.h"
#include "miscadmin.h"
#include "utils/memutils.h"
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

#include "pgx-lower/execution/jit_execution_interface.h"

#include <mlir/InitAllPasses.h>
#include <chrono>

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
}

namespace mlir_runner {

bool setupMLIRContextForJIT(::mlir::MLIRContext& context);
bool runCompleteLoweringPipeline(::mlir::ModuleOp module);
bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest);

#ifdef POSTGRESQL_EXTENSION
auto run_mlir_with_dest_receiver(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext, DestReceiver* dest)
    -> bool {
    if (!plannedStmt || !estate || !dest) {
        auto error = pgx_lower::ErrorManager::postgresqlError("Null parameters provided to MLIR runner with "
                                                              "DestReceiver");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }

    try {
        ::mlir::MLIRContext context;
        if (!setupMLIRContextForJIT(context)) {
            return false;
        }

        // Phase 1: PostgreSQL AST to RelAlg translation
        auto translator = postgresql_ast::create_postgresql_ast_translator(context);
        if (!translator) {
            PGX_ERROR("Failed to create PostgreSQL AST translator");
            return false;
        }

        auto module = translator->translate_query(plannedStmt);
        if (!module) {
            PGX_ERROR("Failed to translate PostgreSQL AST to RelAlg MLIR");
            return false;
        }

        // Verify the generated module
        auto verifyResult = mlir::verify(*module);
        if (failed(verifyResult)) {
            // TODO: Swap this for the dump with stats function
            // Print more detailed error information
            std::string errorString;
            llvm::raw_string_ostream errorStream(errorString);
            module->print(errorStream);

            // Debug: Print the generated MLIR before verification
            {
                std::string mlirString;
                llvm::raw_string_ostream stream(mlirString);
                module->print(stream);
                PGX_LOG(JIT, DEBUG, "Generated RelAlg MLIR:\n%s", mlirString.c_str());
            }
            PGX_ERROR("Initial RelAlg MLIR module verification failed. Module content:\n%s", errorString.c_str());
            return false;
        }

        if (!module) {
            PGX_ERROR("Module is null after AST translation");
            return false;
        }

        // Phase 2-3: Run complete lowering pipeline with PostgreSQL safety wrapper
        bool pipelineSuccess = false;
#ifndef BUILDING_UNIT_TESTS
        PG_TRY();
        {
#endif
            try {
                runCompleteLoweringPipeline(*module);
                pipelineSuccess = true;
            } catch (const std::exception& e) {
                PGX_ERROR("MLIR pipeline exception: %s", e.what());
#ifndef BUILDING_UNIT_TESTS
                ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), 
                               errmsg("MLIR lowering pipeline failed: %s", e.what())));
#endif
                pipelineSuccess = false;
            }
#ifndef BUILDING_UNIT_TESTS
        }
        PG_CATCH();
        {
            PGX_WARNING("PostgreSQL exception during MLIR pipeline execution");
            pipelineSuccess = false;
            PG_RE_THROW();
        }
        PG_END_TRY();
#endif

        if (!pipelineSuccess) {
            return false;
        }

        // Phase 4: JIT execution
        if (!executeJITWithDestReceiver(*module, estate, dest)) {
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        std::string error_msg(e.what());
        if (error_msg.find("Multi-phase aggregation") != std::string::npos) {
            PGX_LOG(JIT, DEBUG, "Expected fallback: %s", e.what());
            return false;
        } else {
            PGX_ERROR("MLIR runner exception: %s", e.what());
#ifndef BUILDING_UNIT_TESTS
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("MLIR compilation failed: %s", e.what())));
#endif
            return false;
        }
    } catch (...) {
        PGX_ERROR("Unknown error in MLIR runner");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Unknown error during MLIR compilation")));
#endif
        return false;
    }
}
#endif

} // namespace mlir_runner