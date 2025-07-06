#include "postgres/my_executor.h"
#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/query_analyzer.h"
#include "core/error_handling.h"
#include "core/logging.h"

#include "executor/executor.h"

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

// Global context for tuple scanning - used by external function
struct TupleScanContext {
    TableScanDesc scanDesc{};
    TupleDesc tupleDesc{};
    bool hasMore{};
    int64_t currentValue{};
};

// Holds PostgreSQL tuple data with dual access patterns:
// 1. MLIR gets simplified int64 values for computation/control flow
// 2. Output preserves original PostgreSQL tuple with full type fidelity
struct PostgreSQLTuplePassthrough {
    HeapTuple originalTuple; // Complete PostgreSQL tuple (ALL data preserved)
    TupleDesc tupleDesc; // Tuple metadata for PostgreSQL operations

    PostgreSQLTuplePassthrough()
    : originalTuple(nullptr)
    , tupleDesc(nullptr) {}

    ~PostgreSQLTuplePassthrough() {
        if (originalTuple) {
            heap_freetuple(originalTuple);
            originalTuple = nullptr;
        }
    }

    // Return a simple signal that we have a valid tuple
    // MLIR only needs to know "continue iterating" vs "end of table"
    // (All actual data passes through via originalTuple)
    int64_t getIterationSignal() const { return originalTuple ? 1 : 0; }
};

// Forward declare ComputedResultStorage for early usage
struct ComputedResultStorage {
    std::vector<Datum> computedValues;     // Computed expression results
    std::vector<bool> computedNulls;       // Null flags for computed results
    std::vector<Oid> computedTypes;        // PostgreSQL type OIDs for computed values
    int numComputedColumns = 0;            // Number of computed columns in current query
    
    void clear() {
        computedValues.clear();
        computedNulls.clear();
        computedTypes.clear();
        numComputedColumns = 0;
    }
    
    void resize(int numColumns) {
        numComputedColumns = numColumns;
        computedValues.resize(numColumns, 0);
        computedNulls.resize(numColumns, true);
        computedTypes.resize(numColumns, InvalidOid);
    }
    
    void setResult(int columnIndex, Datum value, bool isNull, Oid typeOid) {
        if (columnIndex >= 0 && columnIndex < numComputedColumns) {
            computedValues[columnIndex] = value;
            computedNulls[columnIndex] = isNull;
            computedTypes[columnIndex] = typeOid;
        }
    }
};

// Global variables for tuple processing and computed result storage
static TupleScanContext* g_scan_context = nullptr;
static ComputedResultStorage g_computed_results;

struct TupleStreamer {
    DestReceiver* dest;
    TupleTableSlot* slot;
    bool isActive;
    std::vector<int> selectedColumns; // Column indices to project from original tuple

    TupleStreamer()
    : dest(nullptr)
    , slot(nullptr)
    , isActive(false) {}

    void initialize(DestReceiver* destReceiver, TupleTableSlot* tupleSlot) {
        dest = destReceiver;
        slot = tupleSlot;
        isActive = true;
    }

    void setSelectedColumns(const std::vector<int>& columns) { selectedColumns = columns; }

    auto streamTuple(const int64_t value) const -> bool {
        if (!isActive || !dest || !slot) {
            return false;
        }

        ExecClearTuple(slot);
        slot->tts_values[0] = Int64GetDatum(value);
        slot->tts_isnull[0] = false;
        slot->tts_nvalid = 1;
        ExecStoreVirtualTuple(slot);

        return dest->receiveSlot(slot, dest);
    }

    // Stream the complete PostgreSQL tuple (all columns, all types preserved)
    // This is what actually appears in query results
    auto streamCompletePostgreSQLTuple(const PostgreSQLTuplePassthrough& passthrough) const -> bool {
        if (!isActive || !dest || !slot || !passthrough.originalTuple) {
            return false;
        }

        try {
            // Clear the slot first
            ExecClearTuple(slot);

            // The slot is configured for the result tuple descriptor (selected columns only)
            // We need to extract only the projected columns from the original tuple
            const auto origTupleDesc = passthrough.tupleDesc;
            const auto resultTupleDesc = slot->tts_tupleDescriptor;

            // Project columns: mix of original columns and computed expression results
            for (int i = 0; i < resultTupleDesc->natts; i++) {
                bool isnull = false;

                if (i < selectedColumns.size()) {
                    const int origColumnIndex = selectedColumns[i];
                    
                    if (origColumnIndex >= 0 && origColumnIndex < origTupleDesc->natts) {
                        // Regular column: copy from original tuple
                        // PostgreSQL uses 1-based attribute indexing
                        const auto value =
                            heap_getattr(passthrough.originalTuple, origColumnIndex + 1, origTupleDesc, &isnull);
                        slot->tts_values[i] = value;
                        slot->tts_isnull[i] = isnull;
                    }
                    else if (origColumnIndex == -1 && i < g_computed_results.numComputedColumns) {
                        // Computed expression: use stored result from MLIR execution
                        slot->tts_values[i] = g_computed_results.computedValues[i];
                        slot->tts_isnull[i] = g_computed_results.computedNulls[i];
                    }
                    else {
                        // Fallback: null value
                        slot->tts_values[i] = static_cast<Datum>(0);
                        slot->tts_isnull[i] = true;
                    }
                }
                else {
                    slot->tts_values[i] = static_cast<Datum>(0);
                    slot->tts_isnull[i] = true;
                }
            }

            slot->tts_nvalid = resultTupleDesc->natts;
            ExecStoreVirtualTuple(slot);

            return dest->receiveSlot(slot, dest);
        } catch (...) {
            elog(WARNING, "Exception caught in streamCompletePostgreSQLTuple");
            return false;
        }
    }

    void shutdown() {
        isActive = false;
        dest = nullptr;
        slot = nullptr;
    }
};

// Additional global variables for tuple processing
static TupleStreamer g_tuple_streamer;
static PostgreSQLTuplePassthrough g_current_tuple_passthrough;

extern "C" int64_t get_next_tuple() {
    if (!g_scan_context) {
        return -1;
    }

    const auto tuple = heap_getnext(g_scan_context->scanDesc, ForwardScanDirection);
    if (tuple == nullptr) {
        g_scan_context->hasMore = false;
        return -2;
    }

    bool isNull;
    const auto value = heap_getattr(tuple, 1, g_scan_context->tupleDesc, &isNull);

    if (isNull) {
        return -3;
    }

    const int64_t intValue = DatumGetInt64(value);
    g_scan_context->currentValue = intValue;
    g_scan_context->hasMore = true;

    return intValue;
}

struct PostgreSQLTableHandle {
    Relation rel;
    TableScanDesc scanDesc;
    TupleDesc tupleDesc;
    bool isOpen;
};

extern "C" void* open_postgres_table(const char* tableName) {
    try {
        if (!g_scan_context) {
            return nullptr;
        }

        auto* handle = new PostgreSQLTableHandle();
        handle->scanDesc = g_scan_context->scanDesc;
        handle->tupleDesc = g_scan_context->tupleDesc;
        handle->rel = nullptr;
        handle->isOpen = true;

        return handle;
    } catch (...) {
        return nullptr;
    }
}

// MLIR Interface: Read next tuple for iteration control
// Returns: 1 = "we have a tuple", -2 = "end of table"
// Side effect: Preserves COMPLETE PostgreSQL tuple for later streaming
// Architecture: MLIR just iterates, PostgreSQL handles all data types
extern "C" int64_t read_next_tuple_from_table(void* tableHandle) {
    if (!tableHandle) {
        return -1;
    }

    const auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);
    if (!handle->isOpen || !handle->scanDesc) {
        return -1;
    }

    const auto tuple = heap_getnext(handle->scanDesc, ForwardScanDirection);
    if (tuple == nullptr) {
        // End of table reached - return 0 to terminate MLIR loop
        return 0;
    }

    // Clean up previous tuple if it exists
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    // Preserve the COMPLETE tuple (all columns, all types) for output
    g_current_tuple_passthrough.originalTuple = heap_copytuple(tuple);
    g_current_tuple_passthrough.tupleDesc = handle->tupleDesc;

    // Return signal: "we have a tuple" (MLIR only uses this for iteration control)
    return g_current_tuple_passthrough.getIterationSignal();
}

extern "C" void close_postgres_table(void* tableHandle) {
    if (!tableHandle) {
        return;
    }

    auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);
    handle->isOpen = false;
    delete handle;
}

// MLIR Interface: Stream complete PostgreSQL tuple to output
// The 'value' parameter is ignored - it's just MLIR's iteration signal
extern "C" auto add_tuple_to_result(const int64_t value) -> bool {
    // Stream the complete PostgreSQL tuple (all data types preserved)
    return g_tuple_streamer.streamCompletePostgreSQLTuple(g_current_tuple_passthrough);
}

// Typed field access functions for PostgreSQL dialect
extern "C" int32_t get_int_field(void* tuple_handle, const int32_t field_index, bool* is_null) {
    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return 0;
    }

    // PostgreSQL uses 1-based attribute indexing
    const int attr_num = field_index + 1;
    if (attr_num > g_current_tuple_passthrough.tupleDesc->natts) {
        *is_null = true;
        return 0;
    }

    bool isnull;
    const auto value =
        heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num, g_current_tuple_passthrough.tupleDesc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0;
    }

    // Convert to int32 based on PostgreSQL type
    const auto atttypid = TupleDescAttr(g_current_tuple_passthrough.tupleDesc, field_index)->atttypid;
    switch (atttypid) {
    case INT2OID: return (int32_t)DatumGetInt16(value);
    case INT4OID: return DatumGetInt32(value);
    case INT8OID: return static_cast<int32_t>(DatumGetInt64(value)); // Truncate to int32
    default: *is_null = true; return 0;
    }
}

extern "C" int64_t get_text_field(void* tuple_handle, const int32_t field_index, bool* is_null) {
    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return 0;
    }

    // PostgreSQL uses 1-based attribute indexing
    const int attr_num = field_index + 1;
    if (attr_num > g_current_tuple_passthrough.tupleDesc->natts) {
        *is_null = true;
        return 0;
    }

    bool isnull;
    const auto value =
        heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num, g_current_tuple_passthrough.tupleDesc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0;
    }

    // For text types, return pointer to the string data
    const auto atttypid = TupleDescAttr(g_current_tuple_passthrough.tupleDesc, field_index)->atttypid;
    switch (atttypid) {
    case TEXTOID:
    case VARCHAROID:
    case CHAROID: {
        auto* textval = DatumGetTextP(value);
        return reinterpret_cast<int64_t>(VARDATA(textval));
    }
    default: *is_null = true; return 0;
    }
}

// MLIR runtime functions for storing computed expression results
extern "C" void store_int_result(int32_t columnIndex, int32_t value, bool isNull) {
    Datum datum = Int32GetDatum(value);
    g_computed_results.setResult(columnIndex, datum, isNull, INT4OID);
}

extern "C" void store_bool_result(int32_t columnIndex, bool value, bool isNull) {
    Datum datum = BoolGetDatum(value);
    g_computed_results.setResult(columnIndex, datum, isNull, BOOLOID);
}

extern "C" void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull) {
    Datum datum = Int64GetDatum(value);
    g_computed_results.setResult(columnIndex, datum, isNull, INT8OID);
}

extern "C" void store_text_result(int32_t columnIndex, const char* value, bool isNull) {
    Datum datum = 0;
    if (!isNull && value != nullptr) {
        datum = CStringGetTextDatum(value);
    }
    g_computed_results.setResult(columnIndex, datum, isNull, TEXTOID);
}

extern "C" void prepare_computed_results(int32_t numColumns) {
    g_computed_results.resize(numColumns);
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
            foreach(lc, targetList) {
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
            } else {
                // Use original table columns (SELECT *)
                selectedColumns = {0}; // First column
                logger.notice("Configured for table column results");
            }
        } else {
            // Fallback: assume first column
            selectedColumns = {0};
        }
    } else {
        // Fallback: assume first column  
        selectedColumns = {0};
    }
    
    g_tuple_streamer.setSelectedColumns(selectedColumns);
    
    // Use the new AST-based MLIR translation
    const auto mlir_success = mlir_runner::run_mlir_postgres_ast_translation(
        const_cast<PlannedStmt*>(stmt), logger);
    
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

// DEPRECATED: This function is replaced by run_mlir_with_ast_translation
// Keeping as stub to avoid compilation errors
bool run_mlir_with_tuple_scan(const TableScanDesc scanDesc, const TupleDesc tupleDesc, const QueryDesc* queryDesc) {
    elog(WARNING, "DEPRECATED: run_mlir_with_tuple_scan called - should use AST translation instead");
    return false; // Always return false to force use of new path
}
auto MyCppExecutor::execute(const QueryDesc* plan) -> bool {
    // Initialize PostgreSQL error handler if not already set
    if (!pgx_lower::ErrorManager::getHandler()) {
        pgx_lower::ErrorManager::setHandler(std::make_unique<pgx_lower::PostgreSQLErrorHandler>());
    }

    PGX_DEBUG("LLVM version: " + std::to_string(LLVM_VERSION_MAJOR) + "." + 
              std::to_string(LLVM_VERSION_MINOR) + "." + std::to_string(LLVM_VERSION_PATCH));
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
    } else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        // Aggregate query with sequential scan as source
        scanPlan = rootPlan->lefttree;
        PGX_DEBUG("Detected aggregate query with SeqScan source");
    } else {
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
        PGX_INFO("All column types are unsupported (" + std::to_string(unsupportedTypeCount) + 
             "/" + std::to_string(tupdesc->natts) + "), falling back to standard executor");
        table_close(rel, AccessShareLock);
        return false;
    }

    if (unsupportedTypeCount > 0) {
        PGX_DEBUG("Table has " + std::to_string(unsupportedTypeCount) + 
             " unsupported column types out of " + std::to_string(tupdesc->natts) + 
             " total - MLIR will attempt to handle with fallback values");
    }

    const auto scanDesc = table_beginscan(rel, GetActiveSnapshot(), 0, nullptr);

    // Try the new AST-based approach first
    bool mlir_success = run_mlir_with_ast_translation(scanDesc, tupdesc, plan);
    
    // AST translation is the primary and only method now

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return mlir_success;
}
