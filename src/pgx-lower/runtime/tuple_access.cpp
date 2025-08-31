#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/runtime/runtime_templates.h"
#include <vector>
#include "pgx-lower/runtime/PostgreSQLDataSource.h"
#include "pgx-lower/utility/error_handling.h"
#include <array>
#include <cstring>
#include <memory>
#include <vector>
#include <tcop/dest.h>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "access/htup_details.h"
#include "access/heapam.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/relscan.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/numeric.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "storage/lockdefs.h"
#include "utils/memutils.h"
#include "utils/datum.h"
#include "miscadmin.h"
#include "commands/trigger.h"
#include "access/xact.h"
}
#endif

ComputedResultStorage g_computed_results;

std::vector<int> g_field_indices;
TupleStreamer g_tuple_streamer;
PostgreSQLTuplePassthrough g_current_tuple_passthrough;
Oid g_jit_table_oid = InvalidOid;

namespace pgx_lower::runtime {

// Memory context safety check for PostgreSQL operations
bool check_memory_context_safety() {
#ifdef POSTGRESQL_EXTENSION
    // Check if we're in a valid PostgreSQL memory context
    if (!CurrentMemoryContext) {
        PGX_ERROR("check_memory_context_safety: No current memory context");
        return false;
    }

    // Check if the context is valid (not NULL and has valid methods)
    if (CurrentMemoryContext->methods == NULL) {
        PGX_ERROR("check_memory_context_safety: Invalid memory context - no methods");
        return false;
    }

    // Verify the context hasn't been reset/deleted by checking its type field
    if (CurrentMemoryContext->type != T_AllocSetContext && CurrentMemoryContext->type != T_SlabContext
        && CurrentMemoryContext->type != T_GenerationContext)
    {
        PGX_ERROR("check_memory_context_safety: Unknown memory context type");
        return false;
    }

    // If we're in ErrorContext, just log a debug message but allow operations
    if (CurrentMemoryContext == ErrorContext) {
        PGX_LOG(RUNTIME, DEBUG, "check_memory_context_safety: Operating in error recovery context");
    }

    return true;
#else
    // In unit tests, always return true
    return true;
#endif
}

// ============================================================================
// Runtime system implementation
// ============================================================================

template<typename T>
void table_builder_add(void* builder, bool is_valid, T value) {
    auto* tb = static_cast<TableBuilder*>(builder);
    bool is_null = !is_valid;

    if (tb) {
        // Expand storage if needed
        if (tb->current_column_index >= g_computed_results.numComputedColumns) {
            int newSize = tb->current_column_index + 1;
            PGX_LOG(RUNTIME,
                    DEBUG,
                    "table_builder_add: expanding computed results storage from %d to %d columns",
                    g_computed_results.numComputedColumns,
                    newSize);
            prepare_computed_results(newSize);
        }

        // Store with proper type
        PGX_LOG(RUNTIME,
                DEBUG,
                "table_builder_add: storing value at column index %d with is_null=%s",
                tb->current_column_index,
                is_null ? "true" : "false");

        Datum datum = toDatum<T>(value);
        g_computed_results.setResult(tb->current_column_index, datum, is_null, getTypeOid<T>());
        tb->current_column_index++;

        // Update total columns tracking
        if (tb->current_column_index > tb->total_columns) {
            tb->total_columns = tb->current_column_index;
        }
    }
    else {
        PGX_LOG(RUNTIME, DEBUG, "table_builder_add: null builder, using fallback column 0");
        Datum datum = toDatum<T>(value);
        g_computed_results.setResult(0, datum, is_null, getTypeOid<T>());
    }
}

// Explicit instantiations for all supported types
template void table_builder_add<int8_t>(void*, bool, int8_t);
template void table_builder_add<int16_t>(void*, bool, int16_t);
template void table_builder_add<int32_t>(void*, bool, int32_t);
template void table_builder_add<int64_t>(void*, bool, int64_t);
template void table_builder_add<bool>(void*, bool, bool);
template void table_builder_add<float>(void*, bool, float);
template void table_builder_add<double>(void*, bool, double);

// Special handling for strings (VarLen32)
template<>
void table_builder_add<::runtime::VarLen32>(void* builder, bool is_valid, ::runtime::VarLen32 value) {
    auto* tb = static_cast<TableBuilder*>(builder);
    bool is_null = !is_valid;

    if (tb) {
        // Expand storage if needed
        if (tb->current_column_index >= g_computed_results.numComputedColumns) {
            prepare_computed_results(tb->current_column_index + 1);
        }

        if (!is_null && value.getLen() > 0) {
            // Memory context switch for strings
            MemoryContext oldContext = CurrentMemoryContext;
            MemoryContextSwitchTo(CurTransactionContext);

            // Convert VarLen32 to PostgreSQL text
            text* textval = cstring_to_text_with_len(reinterpret_cast<const char*>(value.getPtr()), value.getLen());
            Datum datum = PointerGetDatum(textval);
            g_computed_results.setResult(tb->current_column_index, datum, is_null, TEXTOID);

            MemoryContextSwitchTo(oldContext);
        }
        else {
            g_computed_results.setResult(tb->current_column_index, 0, true, TEXTOID);
        }

        tb->current_column_index++;

        if (tb->current_column_index > tb->total_columns) {
            tb->total_columns = tb->current_column_index;
        }
    }
}

// Template implementation for field extraction
template<typename T>
T extract_field(int32_t field_index, bool* is_null) {
    // Check memory context safety
    if (!check_memory_context_safety()) {
        PGX_ERROR("extract_field: Memory context unsafe for PostgreSQL operations");
        *is_null = true;
        return T{};
    }

    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return T{};
    }

    // PostgreSQL uses 1-based attribute indexing
    const int attr_num = field_index + 1;
    if (attr_num > g_current_tuple_passthrough.tupleDesc->natts) {
        *is_null = true;
        return T{};
    }

    bool isnull;
    Datum value =
        heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num, g_current_tuple_passthrough.tupleDesc, &isnull);
    *is_null = isnull;
    if (isnull)
        return T{};

    Oid typeOid = TupleDescAttr(g_current_tuple_passthrough.tupleDesc, field_index)->atttypid;
    return fromDatum<T>(value, typeOid);
}

template int16_t extract_field<int16_t>(int32_t, bool*);
template int32_t extract_field<int32_t>(int32_t, bool*);
template int64_t extract_field<int64_t>(int32_t, bool*);
template bool extract_field<bool>(int32_t, bool*);
template float extract_field<float>(int32_t, bool*);
template double extract_field<double>(int32_t, bool*);

}

struct PostgreSQLTableHandle {
    Relation rel;
    TableScanDesc scanDesc;
    TupleDesc tupleDesc;
    bool isOpen;
};

// Get column position (attnum) for a given table and column name
// Returns 1-based PostgreSQL attnum, or -1 if not found
extern "C" int32_t get_column_attnum(const char* table_name, const char* column_name) {
#ifdef POSTGRESQL_EXTENSION
    if (!table_name || !column_name) {
        return -1;
    }

    // Use the global table OID if available
    if (g_jit_table_oid != InvalidOid) {
        // Open the relation to get its tuple descriptor
        Relation rel = table_open(g_jit_table_oid, AccessShareLock);
        if (!rel) {
            return -1;
        }

        TupleDesc tupdesc = RelationGetDescr(rel);
        int32_t attnum = -1;

        // Search for the column by name
        for (int i = 0; i < tupdesc->natts; i++) {
            Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
            if (!attr->attisdropped && strcmp(NameStr(attr->attname), column_name) == 0) {
                // PostgreSQL attnum is 1-based
                attnum = i + 1;
                break;
            }
        }

        // Close the relation
        table_close(rel, AccessShareLock);

        return attnum;
    }

    return -1;
#else
    // In unit tests, return -1 to trigger fallback
    return -1;
#endif
}

extern "C" void* open_postgres_table(const char* tableName) {
    PGX_LOG(RUNTIME, IO, "open_postgres_table IN: tableName=%s", tableName ? tableName : "NULL");
    PGX_LOG(RUNTIME, DEBUG, "open_postgres_table called with tableName: %s", tableName ? tableName : "NULL");

    // Critical: Check memory context safety before PostgreSQL operations
    if (!pgx_lower::runtime::check_memory_context_safety()) {
        PGX_ERROR("open_postgres_table: Memory context unsafe for PostgreSQL operations");
        return nullptr;
    }

    try {
        PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Creating PostgreSQLTableHandle...");
        auto* handle = new PostgreSQLTableHandle();

        // JIT-managed table access - we need to open the table ourselves
        PGX_LOG(RUNTIME,
                DEBUG,
                "open_postgres_table: JIT-managed table access, opening table: %s",
                tableName ? tableName : "test");

        // Use the table OID passed from the executor
        if (g_jit_table_oid != InvalidOid) {
            Oid tableOid = g_jit_table_oid;
            PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Using table OID: %u", tableOid);

            // Open the table
            handle->rel = table_open(tableOid, AccessShareLock);
            handle->tupleDesc = RelationGetDescr(handle->rel);

            // Create a new scan
            handle->scanDesc = table_beginscan(handle->rel, GetActiveSnapshot(), 0, nullptr);
            handle->isOpen = true;

            PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Successfully opened table with OID %u", tableOid);
        }
        else {
            PGX_ERROR("open_postgres_table: Cannot determine table to open");
            delete handle;
            return nullptr;
        }

        PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Calling heap_rescan...");
        PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: scanDesc=%p, tupleDesc=%p", handle->scanDesc, handle->tupleDesc);
        if (handle->tupleDesc) {
            PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: tupleDesc->natts=%d", handle->tupleDesc->natts);
        }
        if (handle->scanDesc && handle->scanDesc->rs_rd) {
            PGX_LOG(RUNTIME,
                    DEBUG,
                    "open_postgres_table: scanning relation OID=%u, name=%s",
                    RelationGetRelid(handle->scanDesc->rs_rd),
                    RelationGetRelationName(handle->scanDesc->rs_rd));
        }

        // The issue might be snapshot visibility - ensure we see recent INSERT operations
        // Force command counter increment to ensure we see recent changes in the same transaction
        CommandCounterIncrement();

        // Get a fresh snapshot to see all committed changes
        Snapshot currentSnapshot = GetActiveSnapshot();
        if (currentSnapshot) {
            PGX_LOG(RUNTIME,
                    DEBUG,
                    "open_postgres_table: Updating scan with fresh snapshot xmin=%u, xmax=%u",
                    currentSnapshot->xmin,
                    currentSnapshot->xmax);
            // Update the scan's snapshot to see recent changes
            handle->scanDesc->rs_snapshot = currentSnapshot;
        }

        // PostgreSQL 17.5 heap_rescan signature: heap_rescan(scan, key, set_params, allow_strat, allow_sync,
        // allow_pagemode)
        heap_rescan(handle->scanDesc, nullptr, false, false, false, false);

        PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Successfully created handle, returning %p", handle);
        PGX_LOG(RUNTIME, IO, "open_postgres_table OUT: handle=%p", handle);
        return handle;
    } catch (...) {
        PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Exception caught, returning null");
        PGX_LOG(RUNTIME, IO, "open_postgres_table OUT: null (exception)");
        return nullptr;
    }
}

// MLIR Interface: Read next tuple for iteration control
// Returns: 1 = "we have a tuple", -2 = "end of table"
// Side effect: Preserves COMPLETE PostgreSQL tuple for later streaming
// Architecture: MLIR just iterates, PostgreSQL handles all data types
extern "C" int64_t read_next_tuple_from_table(void* tableHandle) {
    PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table IN: tableHandle=%p", tableHandle);
    if (!tableHandle) {
        PGX_LOG(RUNTIME, DEBUG, "read_next_tuple_from_table: tableHandle is null");
        PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: -1 (null handle)");
        return -1;
    }

    // Critical: Check memory context safety before PostgreSQL heap operations
    if (!pgx_lower::runtime::check_memory_context_safety()) {
        PGX_ERROR("read_next_tuple_from_table: Memory context unsafe for PostgreSQL operations");
        return -1;
    }

    const auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);
    if (!handle->isOpen || !handle->scanDesc) {
        PGX_LOG(RUNTIME, DEBUG, "read_next_tuple_from_table: handle not open or scanDesc is null");
        PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: -1 (invalid handle)");
        return -1;
    }

    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: About to call heap_getnext with scanDesc=%p", handle->scanDesc);
    PGX_LOG(RUNTIME,
            TRACE,
            "read_next_tuple_from_table: scanDesc->rs_rd=%p, snapshot=%p",
            handle->scanDesc->rs_rd,
            handle->scanDesc->rs_snapshot);

    HeapTuple tuple = nullptr;
    try {
        // Add PostgreSQL error handling around heap_getnext
        PG_TRY();
        {
            tuple = heap_getnext(handle->scanDesc, ForwardScanDirection);
            PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: heap_getnext completed, tuple=%p", tuple);
        }
        PG_CATCH();
        {
            PGX_ERROR("read_next_tuple_from_table: heap_getnext threw PostgreSQL exception");
            PG_RE_THROW();
        }
        PG_END_TRY();
    } catch (const std::exception& e) {
        PGX_ERROR("read_next_tuple_from_table: heap_getnext threw C++ exception: %s", e.what());
        return -1;
    } catch (...) {
        PGX_ERROR("read_next_tuple_from_table: heap_getnext threw unknown exception");
        return -1;
    }

    if (tuple == nullptr) {
        PGX_LOG(RUNTIME, DEBUG, "read_next_tuple_from_table: heap_getnext returned null - end of table");
        // End of table reached - return 0 to terminate MLIR loop
        PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: 0 (end of table)");
        return 0;
    }

    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: About to process tuple, cleaning up previous tuple");
    // Clean up previous tuple if it exists
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: About to copy tuple with heap_copytuple");
    // Preserve the COMPLETE tuple (all columns, all types) for output
    g_current_tuple_passthrough.originalTuple = heap_copytuple(tuple);
    g_current_tuple_passthrough.tupleDesc = handle->tupleDesc;

    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: Tuple copied successfully, getting iteration signal");
    // Return signal: "we have a tuple" (MLIR only uses this for iteration control)
    auto result = g_current_tuple_passthrough.getIterationSignal();
    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: Returning iteration signal: %ld", result);
    PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: %ld", result);
    return result;
}

extern "C" void close_postgres_table(void* tableHandle) {
    PGX_LOG(RUNTIME, IO, "close_postgres_table IN: tableHandle=%p", tableHandle);
    PGX_LOG(RUNTIME, DEBUG, "close_postgres_table called with handle: %p", tableHandle);
    if (!tableHandle) {
        return;
    }

    auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);

    // If we opened the table ourselves (JIT-managed), close it properly
    if (handle->rel) {
        PGX_LOG(RUNTIME, DEBUG, "close_postgres_table: Closing JIT-managed table scan");
        if (handle->scanDesc) {
            table_endscan(handle->scanDesc);
        }
        table_close(handle->rel, AccessShareLock);
    }

    handle->isOpen = false;
    delete handle;

    // Reset the global table OID for next query
    PGX_LOG(RUNTIME, DEBUG, "close_postgres_table: Resetting g_jit_table_oid from %u to InvalidOid", g_jit_table_oid);
    g_jit_table_oid = InvalidOid;
    PGX_LOG(RUNTIME, IO, "close_postgres_table OUT");
}

// Helper: Copy a datum to PostgreSQL memory based on its type
static Datum copy_datum_to_postgresql_memory(Datum value, Oid typeOid, bool isNull) {
    if (isNull) {
        return value; // NULL values don't need copying
    }

    // Switch on type to determine copy strategy
    switch (typeOid) {
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID:
        // Text types need deep copy using datumCopy
        return datumCopy(value, false, -1);

    case INT2OID:
    case INT4OID:
    case INT8OID:
    case BOOLOID:
    case FLOAT4OID:
    case FLOAT8OID:
        // Scalar types are pass-by-value, safe to use directly
        return value;

    default:
        // For unknown types, attempt conservative copy
        // TODO: Add more type-specific handling as needed
        return value;
    }
}

// Helper: Validate that memory context is safe for PostgreSQL operations
static bool validate_memory_context_safety(const char* operation) {
    if (!pgx_lower::runtime::check_memory_context_safety()) {
        PGX_ERROR("%s: Memory context unsafe for PostgreSQL operations", operation);
        return false;
    }
    return true;
}

// Helper: Stream a tuple to the destination
static bool
stream_tuple_to_destination(TupleTableSlot* slot, DestReceiver* dest, Datum* values, bool* nulls, int numColumns) {
    if (!slot || !dest) {
        PGX_ERROR("stream_tuple_to_destination: Invalid slot or destination");
        return false;
    }

    // Clear the slot first
    ExecClearTuple(slot);

    // Copy values to slot
    for (int i = 0; i < numColumns; i++) {
        slot->tts_values[i] = values[i];
        slot->tts_isnull[i] = nulls[i];
    }

    slot->tts_nvalid = numColumns;
    ExecStoreVirtualTuple(slot);

    // Send to destination
    return dest->receiveSlot(slot, dest);
}

static bool validate_streaming_context() {
    if (!g_tuple_streamer.isActive || !g_tuple_streamer.dest || !g_tuple_streamer.slot) {
        PGX_LOG(RUNTIME, TRACE, "process_computed_results: Tuple streamer not active");
        return false;
    }
    return true;
}

// Helper: Setup memory context for processing
static MemoryContext setup_processing_memory_context(TupleTableSlot* slot) {
    MemoryContext destContext = slot->tts_mcxt ? slot->tts_mcxt : CurrentMemoryContext;

    if (!destContext) {
        PGX_ERROR("process_computed_results: Invalid destination memory context");
        return nullptr;
    }

    MemoryContextSwitchTo(destContext);
    return CurrentMemoryContext;
}

// Helper: Allocate and process column values
static bool allocate_and_process_columns(Datum** processedValues, bool** processedNulls) {
    *processedValues = (Datum*)palloc(g_computed_results.numComputedColumns * sizeof(Datum));
    *processedNulls = (bool*)palloc(g_computed_results.numComputedColumns * sizeof(bool));

    if (!*processedValues || !*processedNulls) {
        PGX_ERROR("process_computed_results: Memory allocation failed");
        return false;
    }

    // Process each column
    for (int i = 0; i < g_computed_results.numComputedColumns; i++) {
        (*processedValues)[i] = copy_datum_to_postgresql_memory(g_computed_results.computedValues[i],
                                                                g_computed_results.computedTypes[i],
                                                                g_computed_results.computedNulls[i]);
        (*processedNulls)[i] = g_computed_results.computedNulls[i];

        PGX_LOG(RUNTIME,
                TRACE,
                "process_computed_results: col[%d] type=%u null=%s",
                i,
                g_computed_results.computedTypes[i],
                (*processedNulls)[i] ? "true" : "false");
    }

    return true;
}

static bool process_computed_results_for_streaming() {
    // Validate streaming context
    if (!validate_streaming_context()) {
        return false;
    }

    auto slot = g_tuple_streamer.slot;
    MemoryContext oldContext = CurrentMemoryContext;

    // Setup memory context for processing
    MemoryContext destContext = setup_processing_memory_context(slot);
    if (!destContext) {
        return false;
    }

    // Allocate and process columns
    Datum* processedValues = nullptr;
    bool* processedNulls = nullptr;

    if (!allocate_and_process_columns(&processedValues, &processedNulls)) {
        MemoryContextSwitchTo(oldContext);
        return false;
    }

    // Stream the processed tuple
    bool result = stream_tuple_to_destination(slot,
                                              g_tuple_streamer.dest,
                                              processedValues,
                                              processedNulls,
                                              g_computed_results.numComputedColumns);

    // Clean up allocated memory
    pfree(processedValues);
    pfree(processedNulls);

    // Restore original memory context
    MemoryContextSwitchTo(oldContext);

    PGX_LOG(RUNTIME, TRACE, "process_computed_results: streaming returned %s", result ? "true" : "false");
    return result;
}

// MLIR Interface: Stream complete PostgreSQL tuple to output
// The 'value' parameter is ignored - it's just MLIR's iteration signal
extern "C" auto add_tuple_to_result(const int64_t value) -> bool {
    PGX_LOG(RUNTIME, IO, "add_tuple_to_result IN: value=%ld", value);
    PGX_LOG(RUNTIME, TRACE, "add_tuple_to_result: called with value=%ld", value);
    PGX_LOG(RUNTIME, TRACE, "add_tuple_to_result: numComputedColumns=%d", g_computed_results.numComputedColumns);

    // Validate memory context safety
    if (!validate_memory_context_safety("add_tuple_to_result")) {
        return false;
    }

    // Check if we have computed results to stream
    if (g_computed_results.numComputedColumns > 0) {
        PGX_LOG(RUNTIME, DEBUG, "add_tuple_to_result: Streaming computed results");
        return process_computed_results_for_streaming();
    }

    PGX_LOG(RUNTIME, TRACE, "add_tuple_to_result: No computed results available");
    PGX_LOG(RUNTIME, IO, "add_tuple_to_result OUT: false (no computed results)");
    return false;
}


extern "C" bool get_bool_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<bool>(field_index, is_null);
}

extern "C" void prepare_computed_results(int32_t numColumns) {
    g_computed_results.resize(numColumns);
}

// Global flag to indicate results are ready for streaming
bool g_jit_results_ready = false;

// Mark results as ready for streaming (called from JIT)
extern "C" void mark_results_ready_for_streaming() {
    PGX_LOG(RUNTIME, IO, "mark_results_ready_for_streaming IN");
    g_jit_results_ready = true;
    PGX_LOG(RUNTIME, DEBUG, "AFTER: g_jit_results_ready = %d", g_jit_results_ready);

    // Add some validation
    PGX_LOG(RUNTIME, TRACE, "Validation: g_computed_results.numComputedColumns = %d", g_computed_results.numComputedColumns);
    if (g_computed_results.numComputedColumns > 0) {
        PGX_LOG(RUNTIME,
                TRACE,
                "Validation: First computed value = %ld",
                DatumGetInt64(g_computed_results.computedValues[0]));
    }
    PGX_LOG(RUNTIME, IO, "mark_results_ready_for_streaming OUT");
}

//===----------------------------------------------------------------------===//
// C-style interface for MLIR JIT compatibility
//===----------------------------------------------------------------------===//

// get_numeric_field needs to be available for both unit tests and extension
extern "C" double get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<double>(field_index, is_null);
}

extern "C" int16_t get_int16_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<int16_t>(field_index, is_null);
}

// Typed field access functions for PostgreSQL dialect
extern "C" int32_t get_int32_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<int32_t>(field_index, is_null);
}

extern "C" int64_t get_int64_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<int64_t>(field_index, is_null);
}

extern "C" float get_float32_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<float>(field_index, is_null);
}

extern "C" double get_float64_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<double>(field_index, is_null);
}

// MLIR wrappers for new field extraction functions
extern "C" int16_t get_int16_field_mlir(int64_t iteration_signal, int32_t field_index) {
    bool is_null;
    return pgx_lower::runtime::extract_field<int16_t>(field_index, &is_null);
}

extern "C" int64_t get_int64_field_mlir(int64_t iteration_signal, int32_t field_index) {
    bool is_null;
    return pgx_lower::runtime::extract_field<int64_t>(field_index, &is_null);
}

extern "C" float get_float32_field_mlir(int64_t iteration_signal, int32_t field_index) {
    bool is_null;
    return pgx_lower::runtime::extract_field<float>(field_index, &is_null);
}

extern "C" double get_float64_field_mlir(int64_t iteration_signal, int32_t field_index) {
    bool is_null;
    return pgx_lower::runtime::extract_field<double>(field_index, &is_null);
}
