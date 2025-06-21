#include "my_executor.h"
#include "mlir_runner.h"
#include "mlir_logger.h"

#include "executor/executor.h"

extern "C" {
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "postgres.h"
#include "tcop/dest.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
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

struct EnhancedTupleValues {
    static constexpr size_t MAX_COLUMNS = 30;  // Increased for complex tables
    
    HeapTuple originalTuple;
    TupleDesc tupleDesc;
    
    int64_t mlirValues[MAX_COLUMNS];
    bool isNull[MAX_COLUMNS];
    Oid columnTypes[MAX_COLUMNS];
    size_t numColumns;
    
    EnhancedTupleValues() : originalTuple(nullptr), tupleDesc(nullptr), numColumns(0) {
        for (size_t i = 0; i < MAX_COLUMNS; i++) {
            mlirValues[i] = 0;
            isNull[i] = true;
            columnTypes[i] = InvalidOid;
        }
    }
    
    ~EnhancedTupleValues() {
        if (originalTuple) {
            heap_freetuple(originalTuple);
            originalTuple = nullptr;
        }
    }
};

struct TupleStreamer {
    DestReceiver* dest;
    TupleTableSlot* slot;
    bool isActive;
    
    TupleStreamer() : dest(nullptr), slot(nullptr), isActive(false) {}
    
    void initialize(DestReceiver* destReceiver, TupleTableSlot* tupleSlot) {
        dest = destReceiver;
        slot = tupleSlot;
        isActive = true;
    }
    
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
    
    bool streamOriginalTuple(const EnhancedTupleValues& enhancedValues) {
        if (!isActive || !dest || !slot || !enhancedValues.originalTuple) {
            return false;
        }
        
        try {
            ExecStoreHeapTuple(enhancedValues.originalTuple, slot, false);  // false = don't free tuple
            
            return dest->receiveSlot(slot, dest);
        } catch (...) {
            elog(WARNING, "Exception caught in streamOriginalTuple");
            return false;
        }
    }
    
    bool streamTupleMultiple(const EnhancedTupleValues& enhancedValues) {
        // Use the new method that preserves original data
        return streamOriginalTuple(enhancedValues);
    }
    
    void shutdown() {
        isActive = false;
        dest = nullptr;
        slot = nullptr;
    }
};

static TupleScanContext* g_scan_context = nullptr;
static TupleStreamer g_tuple_streamer;
static EnhancedTupleValues g_current_tuple;

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

    TupleDesc tupdesc = handle->tupdesc;
    
    if (g_current_tuple.originalTuple) {
        heap_freetuple(g_current_tuple.originalTuple);
        g_current_tuple.originalTuple = nullptr;
    }
    
    g_current_tuple.originalTuple = heap_copytuple(tuple);
    g_current_tuple.tupleDesc = tupdesc;
    g_current_tuple.numColumns = tupdesc->natts;
    
    size_t maxCols = std::min((size_t)tupdesc->natts, EnhancedTupleValues::MAX_COLUMNS);
    
    for (size_t i = 0; i < maxCols; i++) {
        bool isNull;
        Datum value = heap_getattr(tuple, i + 1, tupdesc, &isNull);  // PostgreSQL columns are 1-indexed
        
        g_current_tuple.isNull[i] = isNull;
        g_current_tuple.columnTypes[i] = TupleDescAttr(tupdesc, i)->atttypid;
        
        if (!isNull) {
            Oid columnType = TupleDescAttr(tupdesc, i)->atttypid;
            switch (columnType) {
                case BOOLOID:
                    g_current_tuple.mlirValues[i] = DatumGetBool(value) ? 1 : 0;
                    break;
                case INT2OID:
                    g_current_tuple.mlirValues[i] = DatumGetInt16(value);
                    break;
                case INT4OID:
                    g_current_tuple.mlirValues[i] = DatumGetInt32(value);
                    break;
                case INT8OID:
                    g_current_tuple.mlirValues[i] = DatumGetInt64(value);
                    break;
                case FLOAT4OID:
                    g_current_tuple.mlirValues[i] = (int64_t)DatumGetFloat4(value);
                    break;
                case FLOAT8OID:
                    g_current_tuple.mlirValues[i] = (int64_t)DatumGetFloat8(value);
                    break;
                default:
                    g_current_tuple.mlirValues[i] = 0;
                    elog(LOG, "Column %zu type %u not converted for MLIR processing (original data preserved)", i, columnType);
                    break;
            }
        } else {
            g_current_tuple.mlirValues[i] = 0;
        }
    }

    if (g_current_tuple.numColumns > 0 && !g_current_tuple.isNull[0]) {
        return g_current_tuple.mlirValues[0];
    }
    
    return -3;
}

extern "C" void close_postgres_table(void* tableHandle) {
    if (!tableHandle) {
        return;
    }

    auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);
    handle->isOpen = false;
    delete handle;
}

extern "C" bool add_tuple_to_result(int64_t value) {
    return g_tuple_streamer.streamOriginalTuple(g_current_tuple);
}

bool run_mlir_with_tuple_scan(TableScanDesc scanDesc, TupleDesc tupdesc, const QueryDesc* queryDesc) {
    PostgreSQLLogger logger;

    DestReceiver* dest = queryDesc->dest;
    TupleTableSlot* slot = MakeSingleTupleTableSlot(tupdesc, &TTSOpsHeapTuple);

    dest->rStartup(dest, queryDesc->operation, tupdesc);

    g_tuple_streamer.initialize(dest, slot);

    TupleScanContext scanContext = {scanDesc, tupdesc, true, 0};
    g_scan_context = &scanContext;

    bool mlir_success = mlir_runner::run_mlir_postgres_table_scan("current_table", logger);
    
    // Cleanup
    g_scan_context = nullptr;
    g_tuple_streamer.shutdown();
    
    if (g_current_tuple.originalTuple) {
        heap_freetuple(g_current_tuple.originalTuple);
        g_current_tuple.originalTuple = nullptr;
    }

    dest->rShutdown(dest);

    ExecDropSingleTupleTableSlot(slot);
    
    return mlir_success;
}

bool MyCppExecutor::execute(const QueryDesc* plan) {
    elog(NOTICE, "LLVM version: %d.%d.%d", LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH);
    if (!plan) {
        elog(ERROR, "QueryDesc is null");
        return false;
    }

    elog(NOTICE, "Inside C++ executor! Plan type: %d", plan->operation);
    elog(NOTICE, "Query text: %s", plan->sourceText ? plan->sourceText : "NULL");

    if (plan->operation != CMD_SELECT) {
        elog(NOTICE, "Not a SELECT statement, skipping");
        return false;
    }

    PlannedStmt* stmt = plan->plannedstmt;
    Plan* rootPlan = stmt->planTree;

    if (rootPlan->type != T_SeqScan) {
        elog(NOTICE, "Only simple table scans (SeqScan) are supported in raw mode.");
        return false;
    }

    SeqScan* scan = (SeqScan*)rootPlan;
    RangeTblEntry* rte = (RangeTblEntry*)list_nth(stmt->rtable, scan->scan.scanrelid - 1);
    Relation rel = table_open(rte->relid, AccessShareLock);
    TupleDesc tupdesc = RelationGetDescr(rel);

    int unsupportedTypeCount = 0;
    for (int i = 0; i < tupdesc->natts; i++) {
        Oid columnType = TupleDescAttr(tupdesc, i)->atttypid;
        if (columnType != BOOLOID && columnType != INT2OID && columnType != INT4OID && 
            columnType != INT8OID && columnType != FLOAT4OID && columnType != FLOAT8OID) {
            unsupportedTypeCount++;
        }
    }

    if (unsupportedTypeCount == tupdesc->natts) {
        elog(NOTICE, "All column types are unsupported (%d/%d), falling back to standard executor", 
             unsupportedTypeCount, tupdesc->natts);
        table_close(rel, AccessShareLock);
        return false;
    }
    
    if (unsupportedTypeCount > 0) {
        elog(NOTICE, "Table has %d unsupported column types out of %d total - MLIR will attempt to handle with fallback values", 
             unsupportedTypeCount, tupdesc->natts);
    }

    TableScanDesc scanDesc = table_beginscan(rel, GetActiveSnapshot(), 0, NULL);

    bool mlir_success = run_mlir_with_tuple_scan(scanDesc, tupdesc, plan);

    table_endscan(scanDesc);
    table_close(rel, AccessShareLock);

    return mlir_success;
}
