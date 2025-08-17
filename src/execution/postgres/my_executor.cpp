#include "execution/postgres/my_executor.h"
#include "execution/mlir_runner.h"
#include "frontend/SQL/query_analyzer.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

#include "runtime/tuple_access.h"

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
EState *CreateExecutorState(void);
void FreeExecutorState(EState *estate);
ExprContext *CreateExprContext(EState *estate);

// Macro for resetting expression context (from executor.h)
#define ResetExprContext(econtext) \
    MemoryContextReset((econtext)->ecxt_per_tuple_memory)
}


// Undefine PostgreSQL macros that conflict with LLVM
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

// PsqlMemoryContextGuard removed - using PostgreSQL-safe PG_TRY/PG_CATCH instead
// RAII patterns can be bypassed by PostgreSQL's longjmp error handling

void logQueryDebugInfo(const PlannedStmt* stmt) {
    PGX_DEBUG("Using PostgreSQL AST translation approach");

    // Debug targetList availability
    PGX_INFO("=== run_mlir_with_ast_translation: Query info ===");
    PGX_INFO("PlannedStmt ptr: " + std::to_string(reinterpret_cast<uintptr_t>(stmt)));
    PGX_INFO("planTree ptr: " + std::to_string(reinterpret_cast<uintptr_t>(stmt->planTree)));
    if (stmt->planTree) {
        PGX_INFO("planTree->targetlist ptr: "
                      + std::to_string(reinterpret_cast<uintptr_t>(stmt->planTree->targetlist)));
        if (stmt->planTree->targetlist) {
            PGX_INFO("targetlist length: " + std::to_string(list_length(stmt->planTree->targetlist)));
        }
        else {
            PGX_INFO("targetlist is NULL!");
        }
    }
}

std::vector<int> analyzeColumnSelection(const PlannedStmt* stmt) {
    // Configure column selection based on query type
    // For SELECT expressions (computed results), use -1 to indicate computed columns
    // For SELECT * (table columns), use 0, 1, 2, etc.
    std::vector<int> selectedColumns;

    // Analyze the planned statement to determine if we have computed expressions
    if (stmt->rtable && list_length(stmt->rtable) > 0) {
        auto* rte = static_cast<RangeTblEntry*>(linitial(stmt->rtable));
        if (rte && stmt->planTree && stmt->planTree->targetlist) {
            auto* targetList = stmt->planTree->targetlist;

            // Check if target list contains expressions (not just simple Vars)
            bool hasComputedExpressions = false;
            ListCell* lc;
            foreach (lc, targetList) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && !tle->resjunk && tle->expr) {
                    // Check if this is a computed expression (not just a simple Var)
                    if (nodeTag(tle->expr) != T_Var) {
                        hasComputedExpressions = true;
                        break;
                    }
                }
            }

            if (hasComputedExpressions) {
                // Use computed results: -1 indicates to use g_computed_results
                selectedColumns = {-1};
                PGX_INFO("Configured for computed expression results");
            }
            else {
                // For now, treat simple SELECT * as computed results since minimal control flow
                // uses store_int_result which populates g_computed_results
                // TODO Phase 6: Eventually fix this to use table columns directly

                // Count actual number of columns selected
                int numSelectedColumns = 0;
                ListCell* lc2;
                foreach (lc2, targetList) {
                    auto* tle = static_cast<TargetEntry*>(lfirst(lc2));
                    if (tle && !tle->resjunk) {
                        numSelectedColumns++;
                    }
                }

                // Create computed result columns for each selected column
                selectedColumns.clear();
                for (int i = 0; i < numSelectedColumns; i++) {
                    selectedColumns.push_back(-1); // -1 indicates computed result
                }
                PGX_INFO("Configured for table column results via computed storage (temporary solution) - "
                              + std::to_string(numSelectedColumns) + " columns");
            }
        }
        else {
            // Fallback: assume first column
            selectedColumns = {0};
        }
    }
    else {
        // Fallback: assume first column
        selectedColumns = {0};
    }

    return selectedColumns;
}

TupleDesc
setupTupleDescriptor(const PlannedStmt* stmt, const std::vector<int>& selectedColumns) {
    // Create result tuple descriptor based on selected columns count
    const int numResultColumns = selectedColumns.size();
    const auto resultTupleDesc = CreateTemplateTupleDesc(numResultColumns);

    // Configure each column in the result tuple descriptor
    for (int i = 0; i < numResultColumns; i++) {
        const auto resultAttr = TupleDescAttr(resultTupleDesc, i);

        // Default type info
        Oid columnType = INT4OID;
        int typeLen = sizeof(int32);
        bool typeByVal = true;
        char typeAlign = TYPALIGN_INT;

        // Try to get column info from target list
        if (stmt->planTree && stmt->planTree->targetlist && i < list_length(stmt->planTree->targetlist)) {
            ListCell* lc;
            int colIdx = 0;
            foreach (lc, stmt->planTree->targetlist) {
                auto* tle = static_cast<TargetEntry*>(lfirst(lc));
                if (tle && !tle->resjunk) {
                    if (colIdx == i) {
                        // Get column name
                        if (tle->resname) {
                            strncpy(NameStr(resultAttr->attname), tle->resname, NAMEDATALEN - 1);
                            PGX_INFO("Setting column " + std::to_string(i)
                                          + " name to: " + std::string(tle->resname));
                        }
                        else {
                            snprintf(NameStr(resultAttr->attname), NAMEDATALEN, "col%d", i);
                            PGX_INFO("Setting column " + std::to_string(i) + " name to: col" + std::to_string(i));
                        }

                        // Get column type from expression
                        if (tle->expr && nodeTag(tle->expr) == T_Var) {
                            Var* var = reinterpret_cast<Var*>(tle->expr);
                            columnType = var->vartype;

                            // Get type properties
                            int16 typLen;
                            bool typByVal;
                            char typAlign;
                            get_typlenbyvalalign(columnType, &typLen, &typByVal, &typAlign);

                            typeLen = typLen;
                            typeByVal = typByVal;
                            typeAlign = typAlign;

                            PGX_INFO("Column " + std::to_string(i) + " type OID: " + std::to_string(columnType));
                        }
                        break;
                    }
                    colIdx++;
                }
            }
        }
        else {
            // Fallback naming
            snprintf(NameStr(resultAttr->attname), NAMEDATALEN, "col%d", i);
        }

        // Set the type info
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
    // Stream results back to PostgreSQL
    if (mlir_success) {
        PGX_INFO("JIT returned successfully, checking results...");
        // Check if JIT marked results as ready
        extern bool g_jit_results_ready;
        PGX_INFO("g_jit_results_ready = " + std::string(g_jit_results_ready ? "true" : "false"));
        if (g_jit_results_ready) {
            PGX_INFO("JIT execution successful - results already streamed by JIT");
            // The JIT now handles all streaming via add_tuple_to_result in the loop
            // We don't need to stream anything here anymore
            g_jit_results_ready = false; // Reset flag
        }
    }
    return mlir_success;
}


// Helper: Initialize PostgreSQL execution resources
static bool initializeExecutionResources(EState** estate, ExprContext** econtext, MemoryContext* old_context) {
    // Create EState for proper PostgreSQL memory context hierarchy
    *estate = CreateExecutorState();
    if (!*estate) {
        PGX_ERROR("Failed to create EState");
        return false;
    }
    
    // Switch to per-query memory context
    *old_context = MemoryContextSwitchTo((*estate)->es_query_cxt);
    
    // Create expression context for safe evaluation
    *econtext = CreateExprContext(*estate);
    if (!*econtext) {
        PGX_ERROR("Failed to create ExprContext");
        return false;
    }
    
    PGX_DEBUG("PostgreSQL memory contexts initialized successfully");
    return true;
}

// Helper: Setup result processing infrastructure
static TupleDesc setupResultProcessing(const PlannedStmt* stmt, DestReceiver* dest, 
                                      TupleTableSlot** slot, CmdType operation) {
    // Analyze and configure column selection
    auto selectedColumns = analyzeColumnSelection(stmt);

    // Initialize computed results storage if using computed results path
    if (!selectedColumns.empty() && selectedColumns[0] == -1) {
        g_computed_results.resize(selectedColumns.size());
        PGX_DEBUG("Allocated computed results storage for " + std::to_string(selectedColumns.size()) + " columns");
    }

    // Setup tuple descriptor for results
    TupleDesc resultTupleDesc = setupTupleDescriptor(stmt, selectedColumns);

    // Initialize PostgreSQL result handling
    *slot = MakeSingleTupleTableSlot(resultTupleDesc, &TTSOpsVirtual);
    dest->rStartup(dest, operation, resultTupleDesc);

    g_tuple_streamer.initialize(dest, *slot);
    g_tuple_streamer.setSelectedColumns(selectedColumns);
    
    return resultTupleDesc;
}

// Helper: Clean up PostgreSQL resources
static void cleanupExecutionResources(EState* estate, ExprContext* econtext, 
                                     TupleTableSlot* slot, TupleDesc resultTupleDesc,
                                     DestReceiver* dest, MemoryContext old_context) {
    PGX_DEBUG("Beginning PostgreSQL-safe cleanup");
    
    // Clean up global tuple streamer
    g_tuple_streamer.shutdown();
    
    // Clean up global tuple passthrough if it exists
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }
    
    // Clean up result streaming resources
    if (dest) {
        dest->rShutdown(dest);
    }
    
    if (slot) {
        ExecDropSingleTupleTableSlot(slot);
    }
    
    if (resultTupleDesc) {
        FreeTupleDesc(resultTupleDesc);
    }
    
    // Reset expression context
    if (econtext) {
        ResetExprContext(econtext);
    }
    
    // Restore original memory context
    MemoryContextSwitchTo(old_context);
    
    // Free EState (automatically cleans up econtext and es_query_cxt)
    if (estate) {
        FreeExecutorState(estate);
    }
}

// Helper: Execute MLIR translation phase
static bool executeMLIRTranslation(PlannedStmt* stmt, EState* estate, 
                                  ExprContext* econtext, DestReceiver* dest) {
    bool mlir_success = mlir_runner::run_mlir_with_dest_receiver(
        stmt, estate, econtext, dest);
    
    PGX_INFO("mlir_runner::run_mlir_with_dest_receiver returned "
                  + std::string(mlir_success ? "true" : "false"));
    
    if (!mlir_success) {
        PGX_ERROR("MLIR compilation failed, falling back to PostgreSQL standard execution");
    }
    
    return mlir_success;
}

// Helper: Validate and prepare query for execution
static bool validateAndPrepareQuery(const QueryDesc* queryDesc, const PlannedStmt** stmt) {
    if (!queryDesc || !queryDesc->plannedstmt) {
        PGX_ERROR("Invalid QueryDesc or PlannedStmt");
        return false;
    }
    
    *stmt = queryDesc->plannedstmt;
    logQueryDebugInfo(*stmt);
    return true;
}

// Helper: Initialize execution context
struct ExecutionContext {
    EState* estate = nullptr;
    ExprContext* econtext = nullptr;
    MemoryContext old_context = nullptr;
    TupleTableSlot* slot = nullptr;
    TupleDesc resultTupleDesc = nullptr;
    bool initialized = false;
};

// Helper: Setup execution with resources
static bool setupExecution(ExecutionContext& ctx, const PlannedStmt* stmt, 
                          DestReceiver* dest, CmdType operation) {
    // Initialize execution resources
    if (!initializeExecutionResources(&ctx.estate, &ctx.econtext, &ctx.old_context)) {
        return false;
    }
    
    ctx.initialized = true;
    
    // Setup result processing
    ctx.resultTupleDesc = setupResultProcessing(stmt, dest, &ctx.slot, operation);
    return true;
}

// Helper: Execute with exception handling
static bool executeWithExceptionHandling(ExecutionContext& ctx, PlannedStmt* stmt,
                                        DestReceiver* dest) {
    bool mlir_success = false;
    
    PG_TRY();
    {
        mlir_success = executeMLIRTranslation(stmt, ctx.estate, ctx.econtext, dest);
    }
    PG_CATCH();
    {
        PGX_ERROR("PostgreSQL exception during MLIR execution");
        if (ctx.initialized) {
            cleanupExecutionResources(ctx.estate, ctx.econtext, ctx.slot, 
                                    ctx.resultTupleDesc, dest, ctx.old_context);
        }
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    return mlir_success;
}

bool run_mlir_with_ast_translation(const QueryDesc* queryDesc) {
    PGX_DEBUG("Using PostgreSQL-safe resource management with PG_TRY/PG_CATCH");
    
    // Validate and prepare query
    const PlannedStmt* stmt = nullptr;
    if (!validateAndPrepareQuery(queryDesc, &stmt)) {
        return false;
    }
    
    // Initialize execution context
    ExecutionContext ctx;
    ctx.old_context = CurrentMemoryContext;
    
    // Setup and execute with exception handling
    if (!setupExecution(ctx, stmt, queryDesc->dest, queryDesc->operation)) {
        ereport(ERROR, (errmsg("Failed to initialize execution resources")));
        return false;
    }
    
    // Execute MLIR translation with exception handling
    bool mlir_success = executeWithExceptionHandling(ctx, const_cast<PlannedStmt*>(stmt), 
                                                    queryDesc->dest);
    
    // Handle results and cleanup
    auto final_result = handleMLIRResults(mlir_success);
    cleanupExecutionResources(ctx.estate, ctx.econtext, ctx.slot, ctx.resultTupleDesc, 
                            queryDesc->dest, ctx.old_context);
    
    PGX_INFO("run_mlir_with_ast_translation completed, returning "
                  + std::string(final_result ? "true" : "false"));
    
    return final_result;
}

auto MyCppExecutor::execute(const QueryDesc* plan) -> bool {
    // Initialize PostgreSQL error handler if not already set
    if (!pgx_lower::ErrorManager::getHandler()) {
        pgx_lower::ErrorManager::setHandler(std::make_unique<pgx_lower::PostgreSQLErrorHandler>());
    }

    PGX_DEBUG("LLVM version: " + std::to_string(LLVM_VERSION_MAJOR) + "." + std::to_string(LLVM_VERSION_MINOR) + "."
              + std::to_string(LLVM_VERSION_PATCH));
    if (!plan) {
        const auto error = pgx_lower::ErrorManager::postgresqlError("QueryDesc is null");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }

    PGX_DEBUG("Inside C++ executor! Plan type: " + std::to_string(plan->operation));
    PGX_DEBUG("Query text: " + std::string(plan->sourceText ? plan->sourceText : "NULL"));

    const auto* stmt = plan->plannedstmt;
    const auto capabilities = pgx_lower::QueryAnalyzer::analyzePlan(stmt);

    PGX_DEBUG("Query analysis: " + std::string(capabilities.getDescription()));

    // 📊 ALWAYS log execution tree regardless of compatibility for comprehensive data collection
    PGX_INFO("📊 FORCING tree logging for all queries in comprehensive collection mode");
    pgx_lower::QueryAnalyzer::validateAndLogPlanStructure(stmt);

    if (!capabilities.isMLIRCompatible()) {
        PGX_INFO("Query requires features not yet supported by MLIR");
        return false;
    }

    // Pass null scanDesc since AST translation doesn't use it
    bool mlir_success = run_mlir_with_ast_translation(plan);

    // AST translation is the primary and only method now
    // No table cleanup needed since JIT handled it

    PGX_INFO("MyCppExecutor::execute completed, returning " + std::string(mlir_success ? "true" : "false"));
    return mlir_success;
}
