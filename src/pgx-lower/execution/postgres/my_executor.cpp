#include "pgx-lower/execution/postgres/my_executor.h"
#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/frontend/SQL/query_analyzer.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

namespace mlir_runner {
auto run_mlir_with_dest_receiver(PlannedStmt* planned_stmt, EState* estate, ExprContext* econtext, DestReceiver* dest)
    -> bool;
} // namespace mlir_runner

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
#include "nodes/nodeFuncs.h"
#include "tcop/dest.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"

// Explicit function declarations for EState management
EState* create_executor_state(void);
void free_executor_state(EState* estate);
ExprContext* create_expr_context(EState* estate);

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

void log_query_debug_info(const PlannedStmt* stmt) {
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

std::vector<int> analyze_column_selection(const PlannedStmt* stmt) {
    std::vector<int> selected_columns;

    if (stmt->rtable && list_length(stmt->rtable) > 0) {
        auto* rte = static_cast<RangeTblEntry*>(linitial(stmt->rtable));
        if (rte && stmt->planTree && stmt->planTree->targetlist) {
            auto* target_list = stmt->planTree->targetlist;

            int num_selected_columns = 0;
            ListCell* lc = nullptr;
            foreach (lc, target_list) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && !tle->resjunk) {
                    num_selected_columns++;
                }
            }

            selected_columns.clear();
            for (int i = 0; i < num_selected_columns; i++) {
                selected_columns.push_back(-1);
            }
            PGX_LOG(GENERAL, DEBUG, "Configured for %d result columns", num_selected_columns);
        }
        else {
            selected_columns = {0};
        }
    }
    else {
        selected_columns = {0};
    }

    return selected_columns;
}

TupleDesc setup_tuple_descriptor(const PlannedStmt* stmt, const std::vector<int>& selected_columns) {
    const int NUM_RESULT_COLUMNS = selected_columns.size();
    auto *const RESULT_TUPLE_DESC = CreateTemplateTupleDesc(NUM_RESULT_COLUMNS);

    for (int i = 0; i < NUM_RESULT_COLUMNS; i++) {
        auto *const RESULT_ATTR = TupleDescAttr(RESULT_TUPLE_DESC, i);

        Oid column_type = INT4OID;
        int type_len = sizeof(int32);
        bool type_by_val = true;
        char type_align = TYPALIGN_INT;

        if (stmt->planTree && stmt->planTree->targetlist && i < list_length(stmt->planTree->targetlist)) {
            ListCell* lc = nullptr;
            int col_idx = 0;
            foreach (lc, stmt->planTree->targetlist) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && !tle->resjunk) {
                    if (col_idx == i) {
                        if (tle->resname) {
                            strncpy(NameStr(RESULT_ATTR->attname), tle->resname, NAMEDATALEN - 1);
                            PGX_LOG(GENERAL, DEBUG, "Setting column %d name to: %s", i, tle->resname);
                        }
                        else {
                            snprintf(NameStr(RESULT_ATTR->attname), NAMEDATALEN, "col%d", i);
                            PGX_LOG(GENERAL, DEBUG, "Setting column %d name to: col%d", i, i);
                        }

                        if (tle->expr) {
                            PGX_LOG(GENERAL, DEBUG, "Column %d: Examining tle->expr nodeTag=%d resname=%s",
                                    i, nodeTag(tle->expr), tle->resname ? tle->resname : "NULL");
                            column_type = exprType((Node*)tle->expr);

                            if (column_type == InvalidOid) {
                                PGX_ERROR("Failed to determine type for expression node type: %d", nodeTag(tle->expr));
                                throw std::runtime_error("Failed to determine type for expression node type");
                            }

                            PGX_LOG(GENERAL, DEBUG, "Column %d: exprType returned OID=%u", i, column_type);

                            int16 typ_len = 0;
                            bool typ_by_val = false;
                            char typ_align = 0;
                            get_typlenbyvalalign(column_type, &typ_len, &typ_by_val, &typ_align);

                            type_len = typ_len;
                            type_by_val = typ_by_val;
                            type_align = typ_align;

                            PGX_LOG(GENERAL, DEBUG, "Column %d type OID: %d (expr type: %d)", i, column_type, nodeTag(tle->expr));
                        }
                        break;
                    }
                    col_idx++;
                }
            }
        }
        else {
            snprintf(NameStr(RESULT_ATTR->attname), NAMEDATALEN, "col%d", i);
        }

        RESULT_ATTR->atttypid = column_type;
        RESULT_ATTR->attlen = type_len;
        RESULT_ATTR->attbyval = type_by_val;
        RESULT_ATTR->attalign = type_align;
        RESULT_ATTR->atttypmod = -1;
        RESULT_ATTR->attnotnull = false;
    }

    return RESULT_TUPLE_DESC;
}

bool handle_mlir_results(bool mlir_success) {
    if (mlir_success) {
        PGX_LOG(JIT, DEBUG, "JIT returned successfully, checking results...");
        
        PGX_LOG(JIT, DEBUG, "g_jit_results_ready = %s", g_jit_results_ready ? "true" : "false");
        if (g_jit_results_ready) {
            PGX_LOG(JIT, DEBUG, "JIT execution successful - results already streamed by JIT");
            g_jit_results_ready = false;
        }
    }
    return mlir_success;
}

static bool initialize_execution_resources(EState** estate, ExprContext** econtext, MemoryContext* old_context) {
    *estate = create_executor_state();
    if (!*estate) {
        PGX_ERROR("Failed to create EState");
        return false;
    }

    *old_context = MemoryContextSwitchTo((*estate)->es_query_cxt);

    *econtext = create_expr_context(*estate);
    if (!*econtext) {
        PGX_ERROR("Failed to create ExprContext");
        return false;
    }

    return true;
}

static TupleDesc
setup_result_processing(const PlannedStmt* stmt, DestReceiver* dest, TupleTableSlot** slot, CmdType operation) {
    auto selected_columns = analyze_column_selection(stmt);

    if (!selected_columns.empty() && selected_columns[0] == -1) {
        g_computed_results.resize(selected_columns.size());
    }

    TupleDesc const result_tuple_desc = setup_tuple_descriptor(stmt, selected_columns);

    for (auto i = 0; i < result_tuple_desc->natts; i++) {
        auto *const ATTR = TupleDescAttr(result_tuple_desc, i);
        if (i < g_computed_results.numComputedColumns) {
            g_computed_results.computedTypes[i] = ATTR->atttypid;
            PGX_LOG(GENERAL, DEBUG, "Initialized computed result column %d with type OID %d", i, ATTR->atttypid);
        } else {
            PGX_WARNING("Managed to access a natt out of range");
        }
    }

    *slot = MakeSingleTupleTableSlot(result_tuple_desc, &TTSOpsVirtual);
    PGX_LOG(GENERAL, DEBUG, "Created slot=%p with tupleDesc=%p, tts_nvalid=%d",
            *slot, (*slot)->tts_tupleDescriptor, (*slot)->tts_nvalid);
    dest->rStartup(dest, operation, result_tuple_desc);

    g_tuple_streamer.initialize(dest, *slot);
    g_tuple_streamer.setSelectedColumns(selected_columns);
    PGX_LOG(GENERAL, DEBUG, "Initialized g_tuple_streamer with slot=%p, dest=%p", *slot, dest);

    return result_tuple_desc;
}

static void cleanup_execution_resources(EState* estate,
                                      ExprContext* econtext,
                                      TupleTableSlot* slot,
                                      TupleDesc result_tuple_desc,
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

    if (result_tuple_desc) {
        FreeTupleDesc(result_tuple_desc);
    }

    if (econtext) {
        ResetExprContext(econtext);
    }

    MemoryContextSwitchTo(old_context);

    if (estate) {
        free_executor_state(estate);
    }
}

static bool execute_mlir_translation(PlannedStmt* stmt, EState* estate, ExprContext* econtext, DestReceiver* dest) {
    bool const mlir_success = mlir_runner::run_mlir_with_dest_receiver(stmt, estate, econtext, dest);

    PGX_LOG(GENERAL, DEBUG, "mlir_runner::run_mlir_with_dest_receiver returned %s", mlir_success ? "true" : "false");

    if (!mlir_success) {
        PGX_ERROR("MLIR compilation failed, falling back to PostgreSQL standard execution");
    }

    return mlir_success;
}

static bool validate_and_prepare_query(const QueryDesc* query_desc, const PlannedStmt** stmt) {
    if (!query_desc || !query_desc->plannedstmt) {
        PGX_ERROR("Invalid QueryDesc or PlannedStmt");
        return false;
    }

    *stmt = query_desc->plannedstmt;
    log_query_debug_info(*stmt);
    return true;
}

struct ExecutionContext {
    EState* estate = nullptr;
    ExprContext* econtext = nullptr;
    MemoryContext old_context = nullptr;
    TupleTableSlot* slot = nullptr;
    TupleDesc result_tuple_desc = nullptr;
    bool initialized = false;
};

static bool setup_execution(ExecutionContext& ctx, const PlannedStmt* stmt, DestReceiver* dest, CmdType operation) {
    if (!initialize_execution_resources(&ctx.estate, &ctx.econtext, &ctx.old_context)) {
        return false;
    }

    ctx.initialized = true;

    ctx.result_tuple_desc = setup_result_processing(stmt, dest, &ctx.slot, operation);
    return true;
}

static bool execute_with_exception_handling(ExecutionContext& ctx, PlannedStmt* stmt, DestReceiver* dest) {
    bool mlir_success = false;

    PG_TRY();
    {
        mlir_success = execute_mlir_translation(stmt, ctx.estate, ctx.econtext, dest);
    }
    PG_CATCH();
    {
        PGX_WARNING("PostgreSQL exception during MLIR execution");
        if (ctx.initialized) {
            cleanup_execution_resources(ctx.estate, ctx.econtext, ctx.slot, ctx.result_tuple_desc, dest, ctx.old_context);
        }
        PG_RE_THROW();
    }
    PG_END_TRY();

    return mlir_success;
}

bool run_mlir_with_ast_translation(const QueryDesc* query_desc) {
    const PlannedStmt* stmt = nullptr;
    if (!validate_and_prepare_query(query_desc, &stmt)) {
        return false;
    }

    ExecutionContext ctx;
    ctx.old_context = CurrentMemoryContext;

    if (!setup_execution(ctx, stmt, query_desc->dest, query_desc->operation)) {
        ereport(ERROR, (errmsg("Failed to initialize execution resources")));
        return false;
    }

    bool const mlir_success = execute_with_exception_handling(ctx, const_cast<PlannedStmt*>(stmt), query_desc->dest);

    auto final_result = handle_mlir_results(mlir_success);
    cleanup_execution_resources(ctx.estate, ctx.econtext, ctx.slot, ctx.result_tuple_desc, query_desc->dest, ctx.old_context);

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
    const auto CAPABILITIES = pgx_lower::QueryAnalyzer::analyzePlan(stmt);

    PGX_LOG(GENERAL, DEBUG, "FORCING tree logging for all queries in comprehensive collection mode");
    pgx_lower::QueryAnalyzer::validateAndLogPlanStructure(stmt);
#else
    auto capabilities = pgx_lower::QueryAnalyzer::analyzeForTesting("test query");
#endif

    if (!CAPABILITIES.isMLIRCompatible()) {
        PGX_LOG(GENERAL, DEBUG, "Query requires features not yet supported by MLIR");
        return false;
    }

    elog(NOTICE, "[PGX-LOWER] Routing through PGX_LOWER compilation");
    bool const mlir_success = run_mlir_with_ast_translation(plan);

    PGX_LOG(GENERAL, DEBUG, "MyCppExecutor::execute completed, returning %s", mlir_success ? "true" : "false");
    return mlir_success;
}
