#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/execution/mlir_runtime.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

// AST Translation
#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"

// Need RelAlgDialect + ColumnManager for per-query reset (spec 01).
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"

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

#ifdef InvalidPid
#define PG_INVALID_PID_SAVED InvalidPid
#undef InvalidPid
#endif

#endif

#include "pgx-lower/execution/jit_execution_engine.h"

#include <mlir/InitAllPasses.h>
#include <chrono>

#ifndef BUILDING_UNIT_TESTS
#ifdef PG_RESTRICT_SAVED
#define restrict PG_RESTRICT_SAVED
#undef PG_RESTRICT_SAVED
#endif

#ifdef PG_INVALID_PID_SAVED
#define InvalidPid PG_INVALID_PID_SAVED
#undef PG_INVALID_PID_SAVED
#endif
#endif

namespace mlir_runner {

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
        // Spec 01: shared MLIRContext with dialects / registry / target init
        // pre-loaded once at _PG_init time. Per-query MLIRContext construction
        // was costing 5-20ms of dialect+target setup on every invocation.
        auto& rt = pgx_lower::execution::get_mlir_runtime();
        ::mlir::MLIRContext& context = rt.context;

        // RelAlg's ColumnManager lives on the dialect (now shared across
        // queries). Clear its per-query accumulator state so stale Column
        // pointers from the previous query can't leak into this query's
        // QueryGraphBuilder lookups.
        context.getLoadedDialect<::mlir::relalg::RelAlgDialect>()->getColumnManager().reset();

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
        pgx_lower::log::verify_module_or_throw(*module, "AST Translation", "PostgreSQL AST to RelAlg MLIR verification failed");

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
        PGX_ERROR("MLIR runner exception: %s", e.what());
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("MLIR compilation failed: %s", e.what())));
#endif
        return false;
    } catch (...) {
        PGX_ERROR("Unknown error in MLIR runner");
#ifndef BUILDING_UNIT_TESTS
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Unknown error during MLIR compilation")));
#endif
        return false;
    }
}
#endif

bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
    pgx_lower::execution::JITEngine engine(llvm::CodeGenOptLevel::Default);

    if (!engine.compile(module)) {
        PGX_ERROR("JIT compilation failed");
        return false;
    }

    if (!engine.execute(estate, dest)) {
        PGX_ERROR("JIT execution failed");
        return false;
    }

    return true;
}

} // namespace mlir_runner