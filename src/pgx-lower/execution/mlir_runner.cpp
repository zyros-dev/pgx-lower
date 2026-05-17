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

#ifdef InvalidPid
#define PG_INVALID_PID_SAVED InvalidPid
#undef InvalidPid
#endif

#endif

#include "pgx-lower/execution/jit_execution_engine.h"

#include <mlir/InitAllPasses.h>
#include <atomic>
#include <chrono>

namespace {
// Convenience alias used throughout this file for millisecond durations.
using DMs = std::chrono::duration<double, std::milli>;
} // namespace

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

bool setupMLIRContextForJIT(::mlir::MLIRContext& context);
bool runCompleteLoweringPipeline(::mlir::ModuleOp module);
bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest);

// Timing struct for the three phases measured inside executeJITWithDestReceiverTimed.
// Does NOT include translate_ms or lowering_ms — those are computed as locals in
// run_mlir_with_dest_receiver and combined with this struct at the log-emission site
// to produce the full 5-phase PGXL_PHASE_TIMING line.
// setup_ms covers MLIRContext construction + setupMLIRContextForJIT (14 dialect loads).
// jit_ms covers JITEngine ctor (setup_llvm_target) + register_dialects + engine.compile().
struct JITPhaseTimings {
    double setup_ms = 0.0;
    double jit_ms   = 0.0;
    double exec_ms  = 0.0;
};

// Internal overload: same logic as the public one but fills *timings when non-null.
// When timings != nullptr, the setup_start passed in is the time point captured just before
// MLIRContext construction; setup_ms is computed as (end of setupMLIRContextForJIT - setup_start).
bool executeJITWithDestReceiverTimed(::mlir::ModuleOp module, EState* estate, DestReceiver* dest,
                                     JITPhaseTimings* timings,
                                     std::chrono::steady_clock::time_point setup_start,
                                     std::chrono::steady_clock::time_point setup_end);

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
        // Determine once whether we should collect per-phase timing.
        // Uses should_log so we only pay timer overhead when the log line will actually fire.
        // Zero cost when pgx_lower.log_enable is off (the dominant, default case).
        const bool timing_enabled = pgx_lower::log::should_log(
            pgx_lower::log::Category::GENERAL, pgx_lower::log::Level::IO);

        // Phase 0: MLIRContext construction + dialect registration (setup_ms).
        // Timer must start BEFORE MLIRContext construction — that object and
        // setupMLIRContextForJIT (14 getOrLoadDialect calls + diagnostic handler)
        // are the prime suspects for unmeasured per-query compile cost.
        const auto t0_start = timing_enabled ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};

        ::mlir::MLIRContext context;
        if (!setupMLIRContextForJIT(context)) {
            return false;
        }

        const auto t0_end = timing_enabled ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};

        // Phase 1: PostgreSQL AST to RelAlg translation
        const auto t1_start = timing_enabled ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};

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

        const auto t1_end = timing_enabled ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};

        // Phase 2: Lowering pipeline (RelAlg → DB → DSA → util → Std → LLVM dialect)
        const auto t2_start = timing_enabled ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};

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

        const auto t2_end = timing_enabled ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};

        // Phases 3 + 4: JIT compile then native execution (timed internally when enabled).
        // Pass setup_start/setup_end so executeJITWithDestReceiverTimed can compute setup_ms.
        JITPhaseTimings timings;
        if (!executeJITWithDestReceiverTimed(*module, estate, dest,
                                             timing_enabled ? &timings : nullptr,
                                             t0_start, t0_end)) {
            return false;
        }

        // Emit ONE structured log line per query — only when log_enable is on.
        if (timing_enabled) {
            const double translate_ms = std::chrono::duration_cast<DMs>(t1_end - t1_start).count();
            const double lowering_ms  = std::chrono::duration_cast<DMs>(t2_end - t2_start).count();
            // Use a monotonic query id (simple per-process counter, not persisted across restarts)
            static std::atomic<uint64_t> query_counter{0};
            const uint64_t qid = ++query_counter;
            PGX_LOG(GENERAL, IO,
                    "PGXL_PHASE_TIMING setup_ms=%.3f translate_ms=%.3f lowering_ms=%.3f jit_ms=%.3f exec_ms=%.3f query=%llu",
                    timings.setup_ms, translate_ms, lowering_ms, timings.jit_ms, timings.exec_ms,
                    static_cast<unsigned long long>(qid));
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

// Internal timed variant: fills *timings (phases 0/setup, 3/jit, 4/exec) when non-null.
// setup_start / setup_end are the time points bracketing MLIRContext construction +
// setupMLIRContextForJIT, captured in the caller before/after that work.
// When timings == nullptr the behaviour is identical to the original function —
// the null check is branch-predicted and has zero measurable overhead.
bool executeJITWithDestReceiverTimed(::mlir::ModuleOp module, EState* estate, DestReceiver* dest,
                                     JITPhaseTimings* timings,
                                     std::chrono::steady_clock::time_point setup_start,
                                     std::chrono::steady_clock::time_point setup_end) {
    // Record setup_ms (MLIRContext ctor + setupMLIRContextForJIT) from caller's measurements.
    if (timings) {
        timings->setup_ms = std::chrono::duration_cast<DMs>(setup_end - setup_start).count();
    }

    // Phase 3: LLVM/JIT compile — timer starts BEFORE JITEngine ctor so that
    // setup_llvm_target (called in the ctor) and register_dialects (called inside
    // engine.compile()) are both included in jit_ms.
    const auto t3_start = timings ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};

    pgx_lower::execution::JITEngine engine(llvm::CodeGenOptLevel::Default);

    if (!engine.compile(module)) {
        PGX_ERROR("JIT compilation failed");
        return false;
    }

    if (timings) {
        timings->jit_ms = std::chrono::duration_cast<DMs>(
            std::chrono::steady_clock::now() - t3_start).count();
    }

    // Phase 4: Native execution
    const auto t4_start = timings ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};

    if (!engine.execute(estate, dest)) {
        PGX_ERROR("JIT execution failed");
        return false;
    }

    if (timings) {
        timings->exec_ms = std::chrono::duration_cast<DMs>(
            std::chrono::steady_clock::now() - t4_start).count();
    }

    return true;
}

// Public API: calls the timed variant without timing (timings == nullptr).
// Behaviour is byte-for-byte identical to the original when timing is not requested.
// The setup_start/setup_end values are unused when timings == nullptr.
bool executeJITWithDestReceiver(::mlir::ModuleOp module, EState* estate, DestReceiver* dest) {
    return executeJITWithDestReceiverTimed(module, estate, dest, nullptr,
                                          std::chrono::steady_clock::time_point{},
                                          std::chrono::steady_clock::time_point{});
}

} // namespace mlir_runner
