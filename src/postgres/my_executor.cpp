#include "postgres/my_executor.h"
#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/query_analyzer.h"
#include "core/error_handling.h"
#include "../../include/core/logging.h"

#include "executor/executor.h"
#include "runtime/tuple_access.h"

#include <vector>

extern "C" {
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "postgres.h"
#include "tcop/dest.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/builtins.h"
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

void registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("convert-to-llvm", "Convert MLIR to LLVM dialect", [](mlir::OpPassManager& pm) {
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertSCFToCFPass());
    });
}

bool run_mlir_with_ast_translation(const TableScanDesc scanDesc, const TupleDesc tupleDesc, const QueryDesc* queryDesc) {
    auto logger = PostgreSQLLogger();

    auto* dest = queryDesc->dest;

    // Extract the planned statement for AST translation
    const auto* stmt = queryDesc->plannedstmt;
    if (!stmt) {
        logger.error("PlannedStmt is null");
        return false;
    }

    logger.debug("Using PostgreSQL AST translation approach");

    // Setup global tuple scanning context (same as before)
    TupleScanContext scanContext = {scanDesc, tupleDesc, true, 0};
    g_scan_context = &scanContext;

    // Create result tuple descriptor - for now, use a simple approach
    const auto resultTupleDesc = CreateTemplateTupleDesc(1);
    const auto resultAttr = TupleDescAttr(resultTupleDesc, 0);
    resultAttr->atttypid = INT8OID;
    resultAttr->attlen = sizeof(int64);
    resultAttr->attbyval = true;
    resultAttr->attalign = TYPALIGN_DOUBLE;
    resultAttr->atttypmod = -1;
    resultAttr->attnotnull = false;
    strncpy(NameStr(resultAttr->attname), "result", NAMEDATALEN - 1);

    const auto slot = MakeSingleTupleTableSlot(resultTupleDesc, &TTSOpsVirtual);

    dest->rStartup(dest, queryDesc->operation, resultTupleDesc);

    g_tuple_streamer.initialize(dest, slot);

    // Clear any previous computed results
    g_computed_results.clear();

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
                // Use original table columns (SELECT *)
                // Automatically detect all columns from the table
                if (scanContext.tupleDesc) {
                    for (int i = 0; i < scanContext.tupleDesc->natts; i++) {
                        selectedColumns.push_back(i);
                    }
                    logger.notice("Configured for table column results (" + std::to_string(selectedColumns.size())
                                  + " columns)");
                }
                else {
                    // Fallback: assume first column
                    selectedColumns = {0};
                    logger.notice("Configured for table column results (fallback: 1 column)");
                }
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

    g_tuple_streamer.setSelectedColumns(selectedColumns);

    // Use the new AST-based MLIR translation
    const auto mlir_success = mlir_runner::run_mlir_postgres_ast_translation(const_cast<PlannedStmt*>(stmt), logger);

    // Cleanup (same as before)
    g_scan_context = nullptr;
    g_tuple_streamer.shutdown();

    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    dest->rShutdown(dest);

    ExecDropSingleTupleTableSlot(slot);
    FreeTupleDesc(resultTupleDesc);

    return mlir_success;
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

    if (plan->operation != CMD_SELECT) {
        elog(NOTICE, "Not a SELECT statement, skipping");
        return false;
    }

    // Use query analyzer to determine MLIR compatibility
    const auto* stmt = plan->plannedstmt;
    const auto capabilities = pgx_lower::QueryAnalyzer::analyzePlan(stmt);

    PGX_DEBUG("Query analysis: " + std::string(capabilities.getDescription()));

    if (!capabilities.isMLIRCompatible()) {
        PGX_INFO("Query requires features not yet supported by MLIR");
        return false;
    }

    const auto rootPlan = stmt->planTree;
    Plan* scanPlan = nullptr;

    if (rootPlan->type == T_SeqScan) {
        // Simple sequential scan query
        scanPlan = rootPlan;
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        // Aggregate query with sequential scan as source
        scanPlan = rootPlan->lefttree;
        PGX_DEBUG("Detected aggregate query with SeqScan source");
    }
    else {
        // This should not happen if analyzer is correct, but add safety check
        PGX_ERROR("Query analyzer bug: marked as compatible but not a simple SeqScan or Agg+SeqScan");
        return false;
    }

    const auto scan = reinterpret_cast<SeqScan*>(scanPlan);
    const auto rte = static_cast<RangeTblEntry*>(list_nth(stmt->rtable, scan->scan.scanrelid - 1));
    const auto rel = table_open(rte->relid, AccessShareLock);
    const auto tupdesc = RelationGetDescr(rel);

    int unsupportedTypeCount = 0;
    for (int i = 0; i < tupdesc->natts; i++) {
        const auto columnType = TupleDescAttr(tupdesc, i)->atttypid;
        if (columnType != BOOLOID && columnType != INT2OID && columnType != INT4OID && columnType != INT8OID
            && columnType != FLOAT4OID && columnType != FLOAT8OID)
        {
            unsupportedTypeCount++;
        }
    }

    if (unsupportedTypeCount == tupdesc->natts) {
        PGX_INFO("All column types are unsupported (" + std::to_string(unsupportedTypeCount) + "/"
                 + std::to_string(tupdesc->natts) + "), falling back to standard executor");
        table_close(rel, AccessShareLock);
        return false;
    }

    if (unsupportedTypeCount > 0) {
        PGX_DEBUG("Table has " + std::to_string(unsupportedTypeCount) + " unsupported column types out of "
                  + std::to_string(tupdesc->natts) + " total - MLIR will attempt to handle with fallback values");
    }

    const auto scanDesc = table_beginscan(rel, GetActiveSnapshot(), 0, nullptr);

    // Try the new AST-based approach first
    bool mlir_success = run_mlir_with_ast_translation(scanDesc, tupdesc, plan);

    // AST translation is the primary and only method now

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return mlir_success;
}
