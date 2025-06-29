#include "postgres/my_executor.h"
#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/query_analyzer.h"
#include "core/error_handling.h"

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
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
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
    TupleDesc tupdesc{};
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
    int64_t getIterationSignal() { return originalTuple ? 1 : 0; }
};

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

    bool streamTuple(int64_t value) {
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
    bool streamCompletePostgreSQLTuple(const PostgreSQLTuplePassthrough& passthrough) {
        if (!isActive || !dest || !slot || !passthrough.originalTuple) {
            return false;
        }

        try {
            // Clear the slot first
            ExecClearTuple(slot);

            // The slot is configured for the result tuple descriptor (selected columns only)
            // We need to extract only the projected columns from the original tuple
            TupleDesc origTupleDesc = passthrough.tupleDesc;
            TupleDesc resultTupleDesc = slot->tts_tupleDescriptor;

            // Project only the selected columns from original tuple to result slot
            for (int i = 0; i < resultTupleDesc->natts; i++) {
                bool isnull = false;

                if (i < selectedColumns.size()) {
                    // Map from result column index to original column index
                    int origColumnIndex = selectedColumns[i];
                    if (origColumnIndex >= 0 && origColumnIndex < origTupleDesc->natts) {
                        // PostgreSQL uses 1-based attribute indexing
                        Datum value =
                            heap_getattr(passthrough.originalTuple, origColumnIndex + 1, origTupleDesc, &isnull);
                        slot->tts_values[i] = value;
                        slot->tts_isnull[i] = isnull;
                    }
                    else {
                        slot->tts_values[i] = (Datum)0;
                        slot->tts_isnull[i] = true;
                    }
                }
                else {
                    slot->tts_values[i] = (Datum)0;
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

static TupleScanContext* g_scan_context = nullptr;
static TupleStreamer g_tuple_streamer;
static PostgreSQLTuplePassthrough g_current_tuple_passthrough;

extern "C" int64_t get_next_tuple() {
    if (!g_scan_context) {
        return -1;
    }

    HeapTuple tuple = heap_getnext(g_scan_context->scanDesc, ForwardScanDirection);
    if (tuple == NULL) {
        g_scan_context->hasMore = false;
        return -2;
    }

    bool isNull;
    Datum value = heap_getattr(tuple, 1, g_scan_context->tupdesc, &isNull);

    if (isNull) {
        return -3;
    }

    int64_t intValue = DatumGetInt64(value);
    g_scan_context->currentValue = intValue;
    g_scan_context->hasMore = true;

    return intValue;
}

struct PostgreSQLTableHandle {
    Relation rel;
    TableScanDesc scanDesc;
    TupleDesc tupdesc;
    bool isOpen;
};

extern "C" void* open_postgres_table(const char* tableName) {
    try {
        if (!g_scan_context) {
            return nullptr;
        }

        auto* handle = new PostgreSQLTableHandle();
        handle->scanDesc = g_scan_context->scanDesc;
        handle->tupdesc = g_scan_context->tupdesc;
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

    auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);
    if (!handle->isOpen || !handle->scanDesc) {
        return -1;
    }

    HeapTuple tuple = heap_getnext(handle->scanDesc, ForwardScanDirection);
    if (tuple == NULL) {
        return -2;
    }

    // Clean up previous tuple if it exists
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    // Preserve the COMPLETE tuple (all columns, all types) for output
    g_current_tuple_passthrough.originalTuple = heap_copytuple(tuple);
    g_current_tuple_passthrough.tupleDesc = handle->tupdesc;

    // Return simple signal: "we have a tuple" (MLIR only uses this for iteration control)
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
extern "C" bool add_tuple_to_result(int64_t value) {
    // Stream the complete PostgreSQL tuple (all data types preserved)
    return g_tuple_streamer.streamCompletePostgreSQLTuple(g_current_tuple_passthrough);
}

// Typed field access functions for PostgreSQL dialect
extern "C" int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return 0;
    }

    // PostgreSQL uses 1-based attribute indexing
    int attr_num = field_index + 1;
    if (attr_num > g_current_tuple_passthrough.tupleDesc->natts) {
        *is_null = true;
        return 0;
    }

    bool isnull;
    Datum value =
        heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num, g_current_tuple_passthrough.tupleDesc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0;
    }

    // Convert to int32 based on PostgreSQL type
    Oid atttypid = TupleDescAttr(g_current_tuple_passthrough.tupleDesc, field_index)->atttypid;
    switch (atttypid) {
    case INT2OID: return (int32_t)DatumGetInt16(value);
    case INT4OID: return DatumGetInt32(value);
    case INT8OID: return (int32_t)DatumGetInt64(value); // Truncate to int32
    default: *is_null = true; return 0;
    }
}

extern "C" int64_t get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return 0;
    }

    // PostgreSQL uses 1-based attribute indexing
    int attr_num = field_index + 1;
    if (attr_num > g_current_tuple_passthrough.tupleDesc->natts) {
        *is_null = true;
        return 0;
    }

    bool isnull;
    Datum value =
        heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num, g_current_tuple_passthrough.tupleDesc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0;
    }

    // For text types, return pointer to the string data
    Oid atttypid = TupleDescAttr(g_current_tuple_passthrough.tupleDesc, field_index)->atttypid;
    switch (atttypid) {
    case TEXTOID:
    case VARCHAROID:
    case CHAROID: {
        text* textval = DatumGetTextP(value);
        return reinterpret_cast<int64_t>(VARDATA(textval));
    }
    default: *is_null = true; return 0;
    }
}

bool run_mlir_with_tuple_scan(TableScanDesc scanDesc, TupleDesc tupdesc, const QueryDesc* queryDesc) {
    PostgreSQLLogger logger;

    DestReceiver* dest = queryDesc->dest;

    // Extract target list from the query plan to get selected columns and create result tuple descriptor
    PlannedStmt* stmt = queryDesc->plannedstmt;
    Plan* rootPlan = stmt->planTree;

    // Create a tuple descriptor for the result (selected columns only)
    TupleDesc resultTupleDesc = CreateTemplateTupleDesc(list_length(rootPlan->targetlist));

    // Get the selected column indices from targetlist and build result tuple descriptor
    std::vector<int> selectedColumns;
    int resultAttrNum = 1;
    ListCell* lc;
    foreach (lc, rootPlan->targetlist) {
        TargetEntry* tle = (TargetEntry*)lfirst(lc);
        if (tle->resjunk)
            continue; // Skip junk columns

        if (IsA(tle->expr, Var)) {
            Var* var = (Var*)tle->expr;
            // Convert PostgreSQL 1-based to 0-based indexing
            selectedColumns.push_back(var->varattno - 1);

            // Copy attribute info from source table to result descriptor
            Form_pg_attribute sourceAttr = TupleDescAttr(tupdesc, var->varattno - 1);
            Form_pg_attribute resultAttr = TupleDescAttr(resultTupleDesc, resultAttrNum - 1);

            resultAttr->atttypid = sourceAttr->atttypid;
            resultAttr->atttypmod = sourceAttr->atttypmod;
            resultAttr->attlen = sourceAttr->attlen;
            resultAttr->attbyval = sourceAttr->attbyval;
            resultAttr->attalign = sourceAttr->attalign;
            resultAttr->attnotnull = sourceAttr->attnotnull;

            // Copy the name or use a generated name
            if (tle->resname) {
                strncpy(NameStr(resultAttr->attname), tle->resname, NAMEDATALEN - 1);
                NameStr(resultAttr->attname)[NAMEDATALEN - 1] = '\0';
            }
            else {
                strncpy(NameStr(resultAttr->attname), NameStr(sourceAttr->attname), NAMEDATALEN - 1);
                NameStr(resultAttr->attname)[NAMEDATALEN - 1] = '\0';
            }

            resultAttrNum++;
        }
    }

    TupleTableSlot* slot = MakeSingleTupleTableSlot(resultTupleDesc, &TTSOpsVirtual);

    dest->rStartup(dest, queryDesc->operation, resultTupleDesc);

    g_tuple_streamer.initialize(dest, slot);
    g_tuple_streamer.setSelectedColumns(selectedColumns);

    TupleScanContext scanContext = {scanDesc, tupdesc, true, 0};
    g_scan_context = &scanContext;

    // Use the new typed field access version with actual column information
    bool mlir_success =
        mlir_runner::run_mlir_postgres_typed_table_scan_with_columns("current_table", selectedColumns, logger);

    // Cleanup
    g_scan_context = nullptr;
    g_tuple_streamer.shutdown();

    // Clean up the current tuple
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    dest->rShutdown(dest);

    ExecDropSingleTupleTableSlot(slot);
    FreeTupleDesc(resultTupleDesc);

    return mlir_success;
}

bool MyCppExecutor::execute(const QueryDesc* plan) {
    // Initialize PostgreSQL error handler if not already set
    if (!pgx_lower::ErrorManager::getHandler()) {
        pgx_lower::ErrorManager::setHandler(std::make_unique<pgx_lower::PostgreSQLErrorHandler>());
    }

    elog(NOTICE, "LLVM version: %d.%d.%d", LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH);
    if (!plan) {
        auto error = pgx_lower::ErrorManager::postgresqlError("QueryDesc is null");
        pgx_lower::ErrorManager::reportError(error);
        return false;
    }

    elog(NOTICE, "Inside C++ executor! Plan type: %d", plan->operation);
    elog(NOTICE, "Query text: %s", plan->sourceText ? plan->sourceText : "NULL");

    if (plan->operation != CMD_SELECT) {
        elog(NOTICE, "Not a SELECT statement, skipping");
        return false;
    }

    // Use query analyzer to determine MLIR compatibility
    PlannedStmt* stmt = plan->plannedstmt;
    auto capabilities = pgx_lower::QueryAnalyzer::analyzePlan(stmt);

    elog(NOTICE, "Query analysis: %s", capabilities.getDescription());

    if (!capabilities.isMLIRCompatible()) {
        elog(NOTICE, "Query requires features not yet supported by MLIR");
        return false;
    }

    Plan* rootPlan = stmt->planTree;
    if (rootPlan->type != T_SeqScan) {
        // This should not happen if analyzer is correct, but add safety check
        elog(NOTICE, "Query analyzer bug: marked as compatible but not a SeqScan");
        return false;
    }

    SeqScan* scan = (SeqScan*)rootPlan;
    RangeTblEntry* rte = (RangeTblEntry*)list_nth(stmt->rtable, scan->scan.scanrelid - 1);
    Relation rel = table_open(rte->relid, AccessShareLock);
    TupleDesc tupdesc = RelationGetDescr(rel);

    int unsupportedTypeCount = 0;
    for (int i = 0; i < tupdesc->natts; i++) {
        Oid columnType = TupleDescAttr(tupdesc, i)->atttypid;
        if (columnType != BOOLOID && columnType != INT2OID && columnType != INT4OID && columnType != INT8OID
            && columnType != FLOAT4OID && columnType != FLOAT8OID)
        {
            unsupportedTypeCount++;
        }
    }

    if (unsupportedTypeCount == tupdesc->natts) {
        elog(NOTICE,
             "All column types are unsupported (%d/%d), falling back to standard executor",
             unsupportedTypeCount,
             tupdesc->natts);
        table_close(rel, AccessShareLock);
        return false;
    }

    if (unsupportedTypeCount > 0) {
        elog(NOTICE,
             "Table has %d unsupported column types out of %d total - MLIR will attempt to handle with fallback values",
             unsupportedTypeCount,
             tupdesc->natts);
    }

    TableScanDesc scanDesc = table_beginscan(rel, GetActiveSnapshot(), 0, NULL);

    bool mlir_success = run_mlir_with_tuple_scan(scanDesc, tupdesc, plan);

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return mlir_success;
}
