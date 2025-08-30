#include "pgx-lower/execution/postgres/my_executor.h"
#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/frontend/SQL/query_analyzer.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

namespace mlir_runner {
auto run_mlir_with_dest_receiver(PlannedStmt* plannedStmt, EState* estate, ExprContext* econtext, DestReceiver* dest)
    -> bool;
}

#include "pgx-lower/runtime/tuple_access.h"

#include <vector>
#include <functional>

extern "C" {
#include "postgres.h"
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "executor/executor.h"
#include "executor/execdesc.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/execnodes.h"
#include "tcop/dest.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"

// Explicit function declarations for EState management
EState* CreateExecutorState(void);
void FreeExecutorState(EState* estate);
ExprContext* CreateExprContext(EState* estate);

#define ResetExprContext(econtext) MemoryContextReset((econtext)->ecxt_per_tuple_memory)
}

#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "llvm/Config/llvm-config.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

void logQueryDebugInfo(const PlannedStmt* stmt) {
    PGX_LOG(GENERAL, DEBUG, "=== run_mlir_with_ast_translation: Query info ===");
    PGX_LOG(GENERAL, DEBUG, "PlannedStmt ptr: %p", stmt);
    PGX_LOG(GENERAL, DEBUG, "planTree ptr: %p", stmt->planTree);
    if (stmt->planTree) {
        PGX_LOG(GENERAL, DEBUG, "planTree->targetlist ptr: %p", stmt->planTree->targetlist);
        if (stmt->planTree->targetlist) {
            PGX_LOG(GENERAL, DEBUG, "targetlist length: %d", list_length(stmt->planTree->targetlist));
        }
        else {
            PGX_LOG(GENERAL, DEBUG, "targetlist is NULL!");
        }
    }
}

std::vector<int> analyzeColumnSelection(const PlannedStmt* stmt) {
    std::vector<int> selectedColumns;

    if (stmt->rtable && list_length(stmt->rtable) > 0) {
        auto* rte = static_cast<RangeTblEntry*>(linitial(stmt->rtable));
        if (rte && stmt->planTree && stmt->planTree->targetlist) {
            auto* targetList = stmt->planTree->targetlist;

            bool hasComputedExpressions = false;
            ListCell* lc;
            foreach (lc, targetList) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && !tle->resjunk && tle->expr) {
                    if (nodeTag(tle->expr) != T_Var) {
                        hasComputedExpressions = true;
                        break;
                    }
                }
            }

            if (hasComputedExpressions) {
                selectedColumns = {-1};
                PGX_LOG(GENERAL, DEBUG, "Configured for computed expression results");
            }
            else {
                // uses store_int_result which populates g_computed_results
                // TODO Phase 6: Eventually fix this to use table columns directly

                int numSelectedColumns = 0;
                ListCell* lc2;
                foreach (lc2, targetList) {
                    auto* tle = static_cast<TargetEntry*>(lfirst(lc2));
                    if (tle && !tle->resjunk) {
                        numSelectedColumns++;
                    }
                }

                selectedColumns.clear();
                for (int i = 0; i < numSelectedColumns; i++) {
                    selectedColumns.push_back(-1);
                }
                PGX_LOG(GENERAL, DEBUG, "Configured for table column results via computed storage (temporary solution) - %d columns", numSelectedColumns);
            }
        }
        else {
            selectedColumns = {0};
        }
    }
    else {
        selectedColumns = {0};
    }

    return selectedColumns;
}

TupleDesc setupTupleDescriptor(const PlannedStmt* stmt, const std::vector<int>& selectedColumns) {
    const int numResultColumns = selectedColumns.size();
    const auto resultTupleDesc = CreateTemplateTupleDesc(numResultColumns);

    for (int i = 0; i < numResultColumns; i++) {
        const auto resultAttr = TupleDescAttr(resultTupleDesc, i);

        Oid columnType = INT4OID;
        int typeLen = sizeof(int32);
        bool typeByVal = true;
        char typeAlign = TYPALIGN_INT;

        if (stmt->planTree && stmt->planTree->targetlist && i < list_length(stmt->planTree->targetlist)) {
            ListCell* lc;
            int colIdx = 0;
            foreach (lc, stmt->planTree->targetlist) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && !tle->resjunk) {
                    if (colIdx == i) {
                        if (tle->resname) {
                            strncpy(NameStr(resultAttr->attname), tle->resname, NAMEDATALEN - 1);
                            PGX_LOG(GENERAL, DEBUG, "Setting column %d name to: %s", i, tle->resname);
                        }
                        else {
                            snprintf(NameStr(resultAttr->attname), NAMEDATALEN, "col%d", i);
                            PGX_LOG(GENERAL, DEBUG, "Setting column %d name to: col%d", i, i);
                        }

                        if (tle->expr && nodeTag(tle->expr) == T_Var) {
                            Var* var = reinterpret_cast<Var*>(tle->expr);
                            columnType = var->vartype;

                            int16 typLen;
                            bool typByVal;
                            char typAlign;
                            get_typlenbyvalalign(columnType, &typLen, &typByVal, &typAlign);

                            typeLen = typLen;
                            typeByVal = typByVal;
                            typeAlign = typAlign;

                            PGX_LOG(GENERAL, DEBUG, "Column %d type OID: %d", i, columnType);
                        }
                        break;
                    }
                    colIdx++;
                }
            }
        }
        else {
            snprintf(NameStr(resultAttr->attname), NAMEDATALEN, "col%d", i);
        }

        resultAttr->atttypid = columnType;
        resultAttr->attlen = typeLen;
        resultAttr->attbyval = typeByVal;
        resultAttr->attalign = typeAlign;
        resultAttr->atttypmod = -1;
        resultAttr->attnotnull = false;
    }

    return resultTupleDesc;
}

bool handleMLIRResults(bool mlir_success) {
    if (mlir_success) {
        PGX_LOG(JIT, DEBUG, "JIT returned successfully, checking results...");
        extern bool g_jit_results_ready;
        PGX_LOG(JIT, DEBUG, "g_jit_results_ready = %s", g_jit_results_ready ? "true" : "false");
        if (g_jit_results_ready) {
            PGX_LOG(JIT, DEBUG, "JIT execution successful - results already streamed by JIT");
            g_jit_results_ready = false;
        }
    }
    return mlir_success;
}

static bool initializeExecutionResources(EState** estate, ExprContext** econtext, MemoryContext* old_context) {
    *estate = CreateExecutorState();
    if (!*estate) {
        PGX_ERROR("Failed to create EState");
        return false;
    }

    *old_context = MemoryContextSwitchTo((*estate)->es_query_cxt);

    *econtext = CreateExprContext(*estate);
    if (!*econtext) {
        PGX_ERROR("Failed to create ExprContext");
        return false;
    }

    return true;
}

static TupleDesc
setupResultProcessing(const PlannedStmt* stmt, DestReceiver* dest, TupleTableSlot** slot, CmdType operation) {
    auto selectedColumns = analyzeColumnSelection(stmt);

    if (!selectedColumns.empty() && selectedColumns[0] == -1) {
        g_computed_results.resize(selectedColumns.size());
    }

    TupleDesc resultTupleDesc = setupTupleDescriptor(stmt, selectedColumns);

    *slot = MakeSingleTupleTableSlot(resultTupleDesc, &TTSOpsVirtual);
    dest->rStartup(dest, operation, resultTupleDesc);

    g_tuple_streamer.initialize(dest, *slot);
    g_tuple_streamer.setSelectedColumns(selectedColumns);

    return resultTupleDesc;
}

static void cleanupExecutionResources(EState* estate,
                                      ExprContext* econtext,
                                      TupleTableSlot* slot,
                                      TupleDesc resultTupleDesc,
                                      DestReceiver* dest,
                                      MemoryContext old_context) {
    g_tuple_streamer.shutdown();

    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    if (dest) {
        dest->rShutdown(dest);
    }

    if (slot) {
        ExecDropSingleTupleTableSlot(slot);
    }

    if (resultTupleDesc) {
        FreeTupleDesc(resultTupleDesc);
    }

    if (econtext) {
        ResetExprContext(econtext);
    }

    MemoryContextSwitchTo(old_context);

    if (estate) {
        FreeExecutorState(estate);
    }
}

static bool executeMLIRTranslation(PlannedStmt* stmt, EState* estate, ExprContext* econtext, DestReceiver* dest) {
    bool mlir_success = mlir_runner::run_mlir_with_dest_receiver(stmt, estate, econtext, dest);

    PGX_LOG(GENERAL, DEBUG, "mlir_runner::run_mlir_with_dest_receiver returned %s", mlir_success ? "true" : "false");

    if (!mlir_success) {
        PGX_ERROR("MLIR compilation failed, falling back to PostgreSQL standard execution");
    }

    return mlir_success;
}

static bool validateAndPrepareQuery(const QueryDesc* queryDesc, const PlannedStmt** stmt) {
    if (!queryDesc || !queryDesc->plannedstmt) {
        PGX_ERROR("Invalid QueryDesc or PlannedStmt");
        return false;
    }

    *stmt = queryDesc->plannedstmt;
    logQueryDebugInfo(*stmt);
    return true;
}

struct ExecutionContext {
    EState* estate = nullptr;
    ExprContext* econtext = nullptr;
    MemoryContext old_context = nullptr;
    TupleTableSlot* slot = nullptr;
    TupleDesc resultTupleDesc = nullptr;
    bool initialized = false;
};

static bool setupExecution(ExecutionContext& ctx, const PlannedStmt* stmt, DestReceiver* dest, CmdType operation) {
    if (!initializeExecutionResources(&ctx.estate, &ctx.econtext, &ctx.old_context)) {
        return false;
    }

    ctx.initialized = true;

    ctx.resultTupleDesc = setupResultProcessing(stmt, dest, &ctx.slot, operation);
    return true;
}

static bool executeWithExceptionHandling(ExecutionContext& ctx, PlannedStmt* stmt, DestReceiver* dest) {
    bool mlir_success = false;

    PG_TRY();
    {
        mlir_success = executeMLIRTranslation(stmt, ctx.estate, ctx.econtext, dest);
    }
    PG_CATCH();
    {
        PGX_ERROR("PostgreSQL exception during MLIR execution");
        if (ctx.initialized) {
            cleanupExecutionResources(ctx.estate, ctx.econtext, ctx.slot, ctx.resultTupleDesc, dest, ctx.old_context);
        }
        PG_RE_THROW();
    }
    PG_END_TRY();

    return mlir_success;
}

bool run_mlir_with_ast_translation(const QueryDesc* queryDesc) {
    const PlannedStmt* stmt = nullptr;
    if (!validateAndPrepareQuery(queryDesc, &stmt)) {
        return false;
    }

    ExecutionContext ctx;
    ctx.old_context = CurrentMemoryContext;

    if (!setupExecution(ctx, stmt, queryDesc->dest, queryDesc->operation)) {
        ereport(ERROR, (errmsg("Failed to initialize execution resources")));
        return false;
    }

    bool mlir_success = executeWithExceptionHandling(ctx, const_cast<PlannedStmt*>(stmt), queryDesc->dest);

    auto final_result = handleMLIRResults(mlir_success);
    cleanupExecutionResources(ctx.estate, ctx.econtext, ctx.slot, ctx.resultTupleDesc, queryDesc->dest, ctx.old_context);

    PGX_LOG(GENERAL, DEBUG, "run_mlir_with_ast_translation completed, returning %s", final_result ? "true" : "false");

    return final_result;
}

auto MyCppExecutor::execute(const QueryDesc* plan) -> bool {
    if (!pgx_lower::ErrorManager::getHandler()) {
        pgx_lower::ErrorManager::setHandler(std::make_unique<pgx_lower::PostgreSQLErrorHandler>());
    }

    if (!plan) {
        const auto error = pgx_lower::ErrorManager::postgresqlError("QueryDesc is null");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }

    const auto* stmt = plan->plannedstmt;
#ifdef POSTGRESQL_EXTENSION
    const auto capabilities = pgx_lower::QueryAnalyzer::analyzePlan(stmt);

    PGX_LOG(GENERAL, DEBUG, "FORCING tree logging for all queries in comprehensive collection mode");
    pgx_lower::QueryAnalyzer::validateAndLogPlanStructure(stmt);
#else
    auto capabilities = pgx_lower::QueryAnalyzer::analyzeForTesting("test query");
#endif

    if (!capabilities.isMLIRCompatible()) {
        PGX_LOG(GENERAL, DEBUG, "Query requires features not yet supported by MLIR");
        return false;
    }

    elog(NOTICE, "[PGX-LOWER] Routing through PGX_LOWER compilation");
    bool mlir_success = run_mlir_with_ast_translation(plan);

    PGX_LOG(GENERAL, DEBUG, "MyCppExecutor::execute completed, returning %s", mlir_success ? "true" : "false");
    return mlir_success;
}
