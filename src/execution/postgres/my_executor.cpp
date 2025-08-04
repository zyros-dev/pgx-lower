#include "execution/postgres/my_executor.h"
#include "execution/mlir_runner.h"
#include "execution/mlir_logger.h"
#include "execution/query_analyzer.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

#include "runtime/tuple_access.h"

#include <vector>

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

/*
 * RAII Pattern for PostgreSQL EState Management
 * 
 * The EStateGuard class ensures that PostgreSQL's EState and ExprContext
 * are properly cleaned up in ALL scenarios:
 * - Normal function exit
 * - Exception thrown during MLIR execution
 * - Early return due to error conditions
 * 
 * This follows PostgreSQL extension best practices and prevents
 * memory leaks that could occur with manual cleanup patterns.
 */

// RAII wrapper for PostgreSQL EState management
class EStateGuard {
private:
    EState* estate_;
    ExprContext* econtext_;
    MemoryContext old_context_;
    bool initialized_;
    
public:
    explicit EStateGuard() 
        : estate_(nullptr), econtext_(nullptr), old_context_(CurrentMemoryContext), initialized_(false) {
        
        // Create EState for proper PostgreSQL memory context hierarchy
        estate_ = CreateExecutorState();
        if (!estate_) {
            throw std::runtime_error("Failed to create EState");
        }
        
        // Switch to per-query memory context
        old_context_ = MemoryContextSwitchTo(estate_->es_query_cxt);
        
        // Create expression context for safe evaluation
        econtext_ = CreateExprContext(estate_);
        if (!econtext_) {
            // Restore context and cleanup EState on failure
            MemoryContextSwitchTo(old_context_);
            FreeExecutorState(estate_);
            throw std::runtime_error("Failed to create ExprContext");
        }
        
        initialized_ = true;
        PGX_DEBUG("EStateGuard: Created EState and ExprContext successfully");
    }
    
    ~EStateGuard() {
        if (initialized_) {
            PGX_DEBUG("EStateGuard: Beginning automatic cleanup");
            
            // Reset per-tuple context if econtext exists
            if (econtext_) {
                ResetExprContext(econtext_);
            }
            
            // Restore original memory context
            MemoryContextSwitchTo(old_context_);
            
            // Free EState (automatically cleans up econtext and es_query_cxt)
            if (estate_) {
                FreeExecutorState(estate_);
            }
            
            PGX_DEBUG("EStateGuard: Automatic cleanup completed");
        }
    }
    
    // Accessors
    EState* getEState() const { 
        return initialized_ ? estate_ : nullptr; 
    }
    
    ExprContext* getExprContext() const { 
        return initialized_ ? econtext_ : nullptr; 
    }
    
    bool isValid() const { 
        return initialized_ && estate_ && econtext_; 
    }
    
    // Non-copyable to prevent resource management issues
    EStateGuard(const EStateGuard&) = delete;
    EStateGuard& operator=(const EStateGuard&) = delete;
    
    // Non-movable for simplicity (could be implemented if needed)
    EStateGuard(EStateGuard&&) = delete;
    EStateGuard& operator=(EStateGuard&&) = delete;
};

void logQueryDebugInfo(const PlannedStmt* stmt, PostgreSQLLogger& logger) {
    logger.debug("Using PostgreSQL AST translation approach");

    // Debug targetList availability
    logger.notice("=== run_mlir_with_ast_translation: Query info ===");
    logger.notice("PlannedStmt ptr: " + std::to_string(reinterpret_cast<uintptr_t>(stmt)));
    logger.notice("planTree ptr: " + std::to_string(reinterpret_cast<uintptr_t>(stmt->planTree)));
    if (stmt->planTree) {
        logger.notice("planTree->targetlist ptr: "
                      + std::to_string(reinterpret_cast<uintptr_t>(stmt->planTree->targetlist)));
        if (stmt->planTree->targetlist) {
            logger.notice("targetlist length: " + std::to_string(list_length(stmt->planTree->targetlist)));
        }
        else {
            logger.notice("targetlist is NULL!");
        }
    }
}

std::vector<int> analyzeColumnSelection(const PlannedStmt* stmt, PostgreSQLLogger& logger) {
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
                // Initialize computed results storage for 1 column
                g_computed_results.resize(1);
                logger.notice("Configured for computed expression results");
            }
            else {
                // For now, treat simple SELECT * as computed results since MinimalSubOpToControlFlow
                // uses store_int_result which populates g_computed_results
                // TODO: Eventually fix this to use table columns directly

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
                g_computed_results.resize(numSelectedColumns);
                logger.notice("Configured for table column results via computed storage (temporary solution) - "
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
setupTupleDescriptor(const PlannedStmt* stmt, const std::vector<int>& selectedColumns, PostgreSQLLogger& logger) {
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
                            logger.notice("Setting column " + std::to_string(i)
                                          + " name to: " + std::string(tle->resname));
                        }
                        else {
                            snprintf(NameStr(resultAttr->attname), NAMEDATALEN, "col%d", i);
                            logger.notice("Setting column " + std::to_string(i) + " name to: col" + std::to_string(i));
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

                            logger.notice("Column " + std::to_string(i) + " type OID: " + std::to_string(columnType));
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

bool handleMLIRResults(bool mlir_success, PostgreSQLLogger& logger) {
    // Stream results back to PostgreSQL
    if (mlir_success) {
        logger.notice("JIT returned successfully, checking results...");
        // Check if JIT marked results as ready
        extern bool g_jit_results_ready;
        logger.notice("g_jit_results_ready = " + std::string(g_jit_results_ready ? "true" : "false"));
        if (g_jit_results_ready) {
            logger.notice("JIT execution successful - results already streamed by JIT");
            // The JIT now handles all streaming via add_tuple_to_result in the loop
            // We don't need to stream anything here anymore
            g_jit_results_ready = false; // Reset flag
        }
    }
    return mlir_success;
}

void cleanupMLIRExecution(DestReceiver* dest, TupleTableSlot* slot, TupleDesc resultTupleDesc, PostgreSQLLogger& logger) {
    // Cleanup (same as before)
    logger.notice("Beginning cleanup phase...");
    g_scan_context = nullptr;
    logger.notice("Shutting down tuple streamer...");
    g_tuple_streamer.shutdown();

    if (g_current_tuple_passthrough.originalTuple) {
        logger.notice("Freeing original tuple...");
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    logger.notice("Shutting down destination...");
    dest->rShutdown(dest);

    logger.notice("Dropping slot and freeing tuple descriptor...");
    ExecDropSingleTupleTableSlot(slot);
    FreeTupleDesc(resultTupleDesc);
}

bool validateAndLogPlanStructure(const PlannedStmt* stmt) {
    const auto rootPlan = stmt->planTree;
    Plan* scanPlan = nullptr;

    if (rootPlan->type == T_SeqScan) {
        scanPlan = rootPlan;
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        scanPlan = rootPlan->lefttree;
        PGX_DEBUG("Detected aggregate query with SeqScan source");
    }
    else {
        PGX_ERROR("Query analyzer bug: marked as compatible but not a simple SeqScan or Agg+SeqScan");
        return false;
    }

    const auto scan = reinterpret_cast<SeqScan*>(scanPlan);
    const auto rte = static_cast<RangeTblEntry*>(list_nth(stmt->rtable, scan->scan.scanrelid - 1));

    PGX_DEBUG("Using AST-based translation - JIT will manage table scan");
    PGX_INFO("Table OID: " + std::to_string(rte->relid));

    return true;
}

bool run_mlir_with_ast_translation(const QueryDesc* queryDesc) {
    auto logger = PostgreSQLLogger();
    auto* dest = queryDesc->dest;

    // Extract the planned statement for AST translation
    const auto* stmt = queryDesc->plannedstmt;
    if (!stmt) {
        logger.error("PlannedStmt is null");
        return false;
    }

    // Log query debug information
    logQueryDebugInfo(stmt, logger);

    PGX_DEBUG("Creating EStateGuard for automatic PostgreSQL memory management");
    
    try {
        // RAII: Automatic EState and ExprContext management
        EStateGuard estate_guard;
        
        if (!estate_guard.isValid()) {
            logger.error("Failed to initialize EState and ExprContext");
            return false;
        }
        
        EState* estate = estate_guard.getEState();
        ExprContext* econtext = estate_guard.getExprContext();
        
        PGX_DEBUG("EState and ExprContext initialized successfully via RAII");
        
        // For AST translation, the JIT manages its own table access
        g_scan_context = nullptr;

        // Analyze and configure column selection
        auto selectedColumns = analyzeColumnSelection(stmt, logger);

        // Setup tuple descriptor for results
        auto resultTupleDesc = setupTupleDescriptor(stmt, selectedColumns, logger);

        // Initialize PostgreSQL result handling
        const auto slot = MakeSingleTupleTableSlot(resultTupleDesc, &TTSOpsVirtual);
        dest->rStartup(dest, queryDesc->operation, resultTupleDesc);

        g_tuple_streamer.initialize(dest, slot);
        g_tuple_streamer.setSelectedColumns(selectedColumns);

        // Execute MLIR translation with proper memory contexts
        const auto mlir_success = mlir_runner::run_mlir_with_estate(
            const_cast<PlannedStmt*>(stmt), estate, econtext, logger);
        
        logger.notice("mlir_runner::run_mlir_with_estate returned "
                      + std::string(mlir_success ? "true" : "false"));

        // Handle results
        auto final_result = handleMLIRResults(mlir_success, logger);

        // Cleanup
        cleanupMLIRExecution(dest, slot, resultTupleDesc, logger);

        logger.notice("run_mlir_with_ast_translation completed successfully, returning "
                      + std::string(final_result ? "true" : "false"));
        
        // EStateGuard destructor will automatically handle all cleanup
        return final_result;
        
    } catch (const std::exception& e) {
        logger.error("Exception in RAII EState management: " + std::string(e.what()));
        // EStateGuard destructor will automatically handle cleanup
        return false;
    } catch (...) {
        logger.error("Unknown exception in RAII EState management");
        // EStateGuard destructor will automatically handle cleanup
        return false;
    }
    
    // Note: No manual cleanup needed - RAII handles everything automatically
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

    if (!capabilities.isMLIRCompatible()) {
        PGX_INFO("Query requires features not yet supported by MLIR");
        return false;
    }

    if (!validateAndLogPlanStructure(stmt)) {
        return false;
    }

    // Pass null scanDesc since AST translation doesn't use it
    bool mlir_success = run_mlir_with_ast_translation(plan);

    // AST translation is the primary and only method now
    // No table cleanup needed since JIT handled it

    PGX_INFO("MyCppExecutor::execute completed, returning " + std::string(mlir_success ? "true" : "false"));
    return mlir_success;
}
