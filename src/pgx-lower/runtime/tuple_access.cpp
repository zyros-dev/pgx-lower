#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/runtime/runtime_templates.h"
#include "pgx-lower/runtime/PostgreSQLRuntime.h"  // For runtime::TableBuilder
#include <vector>
#include "pgx-lower/runtime/PostgreSQLDataSource.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging_c.h"

#include <array>
#include <cstring>
#include <memory>
#include <tcop/dest.h>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "catalog/pg_type_d.h"
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
#include "fmgr.h"
#include "access/xact.h"
}
#endif

//==============================================================================
// Global Variables
//==============================================================================

ComputedResultStorage g_computed_results;
std::vector<int> g_field_indices;
TupleStreamer g_tuple_streamer;
PostgreSQLTuplePassthrough g_current_tuple_passthrough;
Oid g_jit_table_oid = InvalidOid;
bool g_jit_results_ready = false;

//==============================================================================
// Memory Context Safety
//==============================================================================

namespace pgx_lower::runtime {

bool check_memory_context_safety() {
#ifdef POSTGRESQL_EXTENSION
    if (!CurrentMemoryContext) {
        PGX_ERROR("check_memory_context_safety: No current memory context");
        return false;
    }

    if (CurrentMemoryContext->methods == NULL) {
        PGX_ERROR("check_memory_context_safety: Invalid memory context - no methods");
        return false;
    }

    if (CurrentMemoryContext->type != T_AllocSetContext && CurrentMemoryContext->type != T_SlabContext
        && CurrentMemoryContext->type != T_GenerationContext)
    {
        PGX_ERROR("check_memory_context_safety: Unknown memory context type");
        return false;
    }

    if (CurrentMemoryContext == ErrorContext) {
        PGX_LOG(RUNTIME, DEBUG, "check_memory_context_safety: Operating in error recovery context");
    }

    return true;
#else
    return true;
#endif
}

template<typename T>
T extract_field(int32_t field_index, bool* is_null) {
    if (!check_memory_context_safety()) {
        PGX_ERROR("extract_field: Memory context unsafe for PostgreSQL operations");
        *is_null = true;
        return T{};
    }

    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return T{};
    }

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

} // namespace pgx_lower::runtime

//==============================================================================
// Template-Based Table Builder
//==============================================================================

struct PostgreSQLTableHandle {
    Relation rel;
    TableScanDesc scanDesc;
    TupleDesc tupleDesc;
    bool isOpen;
};

namespace pgx_lower::runtime {

template<typename T>
void table_builder_add(void* builder, bool is_valid, T value) {
    auto* tb = static_cast<::runtime::TableBuilder*>(builder);
    
    // LLVM JIT may pass non-standard boolean values (e.g., 255 instead of 1)
    // Pretty weird.
    bool normalized_is_valid = (is_valid != 0);
    bool is_null = !normalized_is_valid;

    if (tb) {
        if (tb->current_column_index >= g_computed_results.numComputedColumns) {
            int newSize = tb->current_column_index + 1;
            PGX_LOG(RUNTIME,
                    DEBUG,
                    "table_builder_add: expanding computed results storage from %d to %d columns",
                    g_computed_results.numComputedColumns,
                    newSize);
            prepare_computed_results(newSize);
        }

        // TODO LLVM may pass different representations?
        // - Regular operations (AND/OR): 0 = false, 1 = true  
        // - CmpIOp operations (NOT): 254 = false, 255 = true
        // I believe this is because arith.cmpi casts to i8 internally, which causes the i1 to extend into 0xFF and 0xFE
        // I spent some time attempting to fix it, but it just didn't work. So now we have this:
        // Either way, this also works with the usual boolean values so... it's probably okay? I dunno.
        T normalized_value = value;
        if constexpr (std::is_same_v<T, bool>) {
            unsigned char byte_value = static_cast<unsigned char>(value);
            // Any non-zero value except 254 (CmpIOp false) is true
            // 0 = false, 1 = true, 254 = false, 255 = true, others = true
            normalized_value = (byte_value != 0) && (byte_value != 254);
            PGX_LOG(RUNTIME,
                    TRACE,
                    "table_builder_add<bool>: byte value=0x%02X (%d), normalized=%d",
                    byte_value,
                    byte_value,
                    static_cast<int>(normalized_value));
        }
        
        PGX_LOG(RUNTIME,
                DEBUG,
                "table_builder_add: storing value at column index %d with is_null=%s",
                tb->current_column_index,
                is_null ? "true" : "false");

        Datum datum = is_null ? (Datum)0 : toDatum<T>(normalized_value);
        g_computed_results.setResult(tb->current_column_index, datum, is_null, getTypeOid<T>());
        tb->current_column_index++;
        if (tb->current_column_index > tb->total_columns) {
            tb->total_columns = tb->current_column_index;
        }
    }
    else {
        PGX_LOG(RUNTIME, DEBUG, "table_builder_add: null builder, using fallback column 0");
        // Use normalized is_null value
        Datum datum = is_null ? (Datum)0 : toDatum<T>(value);
        g_computed_results.setResult(0, datum, is_null, getTypeOid<T>());
    }
}

// Explicit instantiations
template void table_builder_add<int8_t>(void*, bool, int8_t);
template void table_builder_add<int16_t>(void*, bool, int16_t);
template void table_builder_add<int32_t>(void*, bool, int32_t);
template void table_builder_add<int64_t>(void*, bool, int64_t);
template void table_builder_add<bool>(void*, bool, bool);
template void table_builder_add<float>(void*, bool, float);
template void table_builder_add<double>(void*, bool, double);

template<>
void table_builder_add<::runtime::VarLen32>(void* builder, bool is_valid, ::runtime::VarLen32 value) {
    auto* tb = static_cast<::runtime::TableBuilder*>(builder);
    bool is_null = !is_valid;

    if (tb) {
        if (tb->current_column_index >= g_computed_results.numComputedColumns) {
            prepare_computed_results(tb->current_column_index + 1);
        }

        if (!is_null && value.getLen() > 0) {
            PGX_LOG(RUNTIME, DEBUG,
                    "VarLen32: len=%u, ptr=%p, first chars: %.10s", 
                    value.getLen(), 
                    value.getPtr(),
                    value.getPtr());
            
            MemoryContext oldContext = CurrentMemoryContext;
            MemoryContextSwitchTo(CurTransactionContext);

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

} // namespace pgx_lower::runtime

//==============================================================================
// Table Management Functions
//==============================================================================

extern "C" int32_t get_column_attnum(const char* table_name, const char* column_name) {
#ifdef POSTGRESQL_EXTENSION
    if (!table_name || !column_name) {
        return -1;
    }

    if (g_jit_table_oid != InvalidOid) {
        Relation rel = table_open(g_jit_table_oid, AccessShareLock);
        if (!rel) {
            return -1;
        }

        TupleDesc tupdesc = RelationGetDescr(rel);
        int32_t attnum = -1;

        for (int i = 0; i < tupdesc->natts; i++) {
            Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
            if (!attr->attisdropped && strcmp(NameStr(attr->attname), column_name) == 0) {
                attnum = i + 1;
                break;
            }
        }

        table_close(rel, AccessShareLock);
        return attnum;
    }

    return -1;
#else
    return -1;
#endif
}

extern "C" int32_t get_all_column_metadata(const char* table_name, ColumnMetadata* metadata, int32_t max_columns) {
#ifdef POSTGRESQL_EXTENSION
    if (table_name == NULL || metadata == NULL || max_columns <= 0) {
        PGX_ERROR("get_all_column_metadata: Invalid parameters");
        return 0;
    }

    if (g_jit_table_oid == InvalidOid) {
        PGX_ERROR("get_all_column_metadata: g_jit_table_oid is invalid");
        return 0;
    }

    Relation rel = table_open(g_jit_table_oid, AccessShareLock);
    if (!rel) {
        PGX_ERROR("get_all_column_metadata: Failed to open table with OID %u", g_jit_table_oid);
        return 0;
    }

    TupleDesc tupdesc = RelationGetDescr(rel);
    int32_t column_count = 0;

    for (int i = 0; i < tupdesc->natts && column_count < max_columns; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        if (!attr->attisdropped) {
            // Copy column name (64 is the size we defined in the struct)
            strncpy(metadata[column_count].name, NameStr(attr->attname), 63);
            metadata[column_count].name[63] = '\0';
            
            // Store type OID and attribute number
            metadata[column_count].type_oid = attr->atttypid;
            metadata[column_count].attnum = i + 1;  // PostgreSQL uses 1-based indexing
            
            PGX_LOG(RUNTIME, DEBUG, "get_all_column_metadata: Column %d: name='%s', type_oid=%d, attnum=%d", 
                    column_count, metadata[column_count].name, 
                    metadata[column_count].type_oid, metadata[column_count].attnum);
            
            column_count++;
        }
    }

    table_close(rel, AccessShareLock);
    
    PGX_LOG(RUNTIME, DEBUG, "get_all_column_metadata: Retrieved metadata for %d columns", column_count);
    return column_count;
#else
    return 0;
#endif
}


extern "C" void* open_postgres_table(const char* tableName) {
    PGX_LOG(RUNTIME, IO, "open_postgres_table IN: tableName=%s", tableName ? tableName : "NULL");
    PGX_LOG(RUNTIME, DEBUG, "open_postgres_table called with tableName: %s", tableName ? tableName : "NULL");

    if (!pgx_lower::runtime::check_memory_context_safety()) {
        PGX_ERROR("open_postgres_table: Memory context unsafe for PostgreSQL operations");
        return nullptr;
    }

    try {
        PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Creating PostgreSQLTableHandle...");
        auto* handle = new PostgreSQLTableHandle();

        PGX_LOG(RUNTIME,
                DEBUG,
                "open_postgres_table: JIT-managed table access, opening table: %s",
                tableName ? tableName : "test");

        if (g_jit_table_oid != InvalidOid) {
            Oid tableOid = g_jit_table_oid;
            PGX_LOG(RUNTIME, DEBUG, "open_postgres_table: Using table OID: %u", tableOid);

            handle->rel = table_open(tableOid, AccessShareLock);
            handle->tupleDesc = RelationGetDescr(handle->rel);

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

        CommandCounterIncrement();

        Snapshot currentSnapshot = GetActiveSnapshot();
        if (currentSnapshot) {
            PGX_LOG(RUNTIME,
                    DEBUG,
                    "open_postgres_table: Updating scan with fresh snapshot xmin=%u, xmax=%u",
                    currentSnapshot->xmin,
                    currentSnapshot->xmax);
            handle->scanDesc->rs_snapshot = currentSnapshot;
        }

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

extern "C" int64_t read_next_tuple_from_table(void* tableHandle) {
    PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table IN: tableHandle=%p", tableHandle);
    if (!tableHandle) {
        PGX_LOG(RUNTIME, DEBUG, "read_next_tuple_from_table: tableHandle is null");
        PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: -1 (null handle)");
        return -1;
    }

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
        PG_TRY();
        {
            tuple = heap_getnext(handle->scanDesc, ForwardScanDirection);
            PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: heap_getnext completed, tuple=%p", tuple);
        }
        PG_CATCH();
        {
            PGX_ERROR("read_next_tuple_from_table: heap_getnext threw PostgreSQL exception");
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
        PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: 0 (end of table)");
        return 0;
    }

    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: About to process tuple, cleaning up previous tuple");
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
    }

    g_current_tuple_passthrough.originalTuple = heap_copytuple(tuple);
    g_current_tuple_passthrough.tupleDesc = handle->tupleDesc;

    PGX_LOG(RUNTIME, TRACE, "read_next_tuple_from_table: Tuple preserved for streaming");
    PGX_LOG(RUNTIME, IO, "read_next_tuple_from_table OUT: 1 (tuple available)");
    return 1;
}

extern "C" void close_postgres_table(void* tableHandle) {
    PGX_LOG(RUNTIME, IO, "close_postgres_table IN: tableHandle=%p", tableHandle);
    PGX_LOG(RUNTIME, DEBUG, "close_postgres_table called with handle: %p", tableHandle);
    if (!tableHandle) {
        return;
    }

    auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);

    if (handle->rel) {
        PGX_LOG(RUNTIME, DEBUG, "close_postgres_table: Closing JIT-managed table scan");
        if (handle->scanDesc) {
            table_endscan(handle->scanDesc);
        }
        table_close(handle->rel, AccessShareLock);
    }

    handle->isOpen = false;
    delete handle;

    PGX_LOG(RUNTIME, DEBUG, "close_postgres_table: Resetting g_jit_table_oid from %u to InvalidOid", g_jit_table_oid);
    g_jit_table_oid = InvalidOid;
    PGX_LOG(RUNTIME, IO, "close_postgres_table OUT");
}

//==============================================================================
// Memory and Streaming Helper Functions
//==============================================================================

static Datum copy_datum_to_postgresql_memory(Datum value, Oid typeOid, bool isNull) {
    if (isNull) {
        return value;
    }

    switch (typeOid) {
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: return datumCopy(value, false, -1);

    case INT2OID:
    case INT4OID:
    case INT8OID:
    case BOOLOID:
    case FLOAT4OID:
    case FLOAT8OID:
    case NUMERICOID:  // For now, treat NUMERIC as pass-through until we have proper support
        return value;

    default:
        PGX_LOG(RUNTIME, DEBUG, "copy_datum_to_postgresql_memory: Unhandled type OID %u, passing through", typeOid);
        return value;
    }
}

static bool validate_memory_context_safety(const char* operation) {
    PGX_IO(RUNTIME);
    if (!pgx_lower::runtime::check_memory_context_safety()) {
        PGX_ERROR("%s: Memory context unsafe for PostgreSQL operations", operation);
        return false;
    }
    return true;
}

static bool
stream_tuple_to_destination(TupleTableSlot* slot, DestReceiver* dest, Datum* values, bool* nulls, int numColumns) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "stream_tuple_to_destination: slot=%p, dest=%p, numColumns=%d", slot, dest, numColumns);
    if (!slot || !dest) {
        PGX_ERROR("stream_tuple_to_destination: Invalid slot or destination");
        return false;
    }

    TupleDesc tupdesc = slot->tts_tupleDescriptor;
    PGX_LOG(RUNTIME, DEBUG, "stream_tuple_to_destination: slot->tts_tupleDescriptor=%p, slot->tts_nvalid=%d, tupdesc->natts=%d",
            tupdesc, slot->tts_nvalid, tupdesc ? tupdesc->natts : -1);

    ExecClearTuple(slot);
    PGX_LOG(RUNTIME, TRACE, "After ExecClearTuple: tts_nvalid=%d", slot->tts_nvalid);

    for (int i = 0; i < numColumns; i++) {
        slot->tts_values[i] = values[i];
        slot->tts_isnull[i] = nulls[i];
        PGX_LOG(RUNTIME, TRACE, "  Column %d: value=%ld, isnull=%d", i, (long)values[i], nulls[i]);
    }

    slot->tts_nvalid = numColumns;
    PGX_LOG(RUNTIME, TRACE, "After setting tts_nvalid=%d", slot->tts_nvalid);

    ExecStoreVirtualTuple(slot);
    PGX_LOG(RUNTIME, TRACE, "After ExecStoreVirtualTuple: tts_nvalid=%d", slot->tts_nvalid);

    PGX_LOG(RUNTIME, DEBUG, "About to call dest->receiveSlot with slot=%p, dest=%p", slot, dest);
    bool result = dest->receiveSlot(slot, dest);
    PGX_LOG(RUNTIME, DEBUG, "dest->receiveSlot returned %d", result);
    return result;
}

static bool validate_streaming_context() {
    PGX_IO(RUNTIME);
    if (!g_tuple_streamer.isActive || !g_tuple_streamer.dest || !g_tuple_streamer.slot) {
        PGX_LOG(RUNTIME, TRACE, "process_computed_results: Tuple streamer not active");
        return false;
    }
    return true;
}

static MemoryContext setup_processing_memory_context(TupleTableSlot* slot) {
    PGX_IO(RUNTIME);
    MemoryContext destContext = slot->tts_mcxt ? slot->tts_mcxt : CurrentMemoryContext;

    if (!destContext) {
        PGX_ERROR("process_computed_results: Invalid destination memory context");
        return nullptr;
    }

    MemoryContextSwitchTo(destContext);
    return CurrentMemoryContext;
}

static bool allocate_and_process_columns(Datum** processedValues, bool** processedNulls) {
    PGX_IO(RUNTIME);
    *processedValues = (Datum*)palloc(g_computed_results.numComputedColumns * sizeof(Datum));
    *processedNulls = (bool*)palloc(g_computed_results.numComputedColumns * sizeof(bool));

    if (!*processedValues || !*processedNulls) {
        PGX_ERROR("process_computed_results: Memory allocation failed");
        return false;
    }

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
    PGX_IO(RUNTIME);
    if (!validate_streaming_context()) {
        return false;
    }

    auto slot = g_tuple_streamer.slot;
    MemoryContext oldContext = CurrentMemoryContext;

    MemoryContext destContext = setup_processing_memory_context(slot);
    if (!destContext) {
        return false;
    }

    Datum* processedValues = nullptr;
    bool* processedNulls = nullptr;

    if (!allocate_and_process_columns(&processedValues, &processedNulls)) {
        MemoryContextSwitchTo(oldContext);
        return false;
    }

    bool result = stream_tuple_to_destination(slot,
                                              g_tuple_streamer.dest,
                                              processedValues,
                                              processedNulls,
                                              g_computed_results.numComputedColumns);

    pfree(processedValues);
    pfree(processedNulls);

    MemoryContextSwitchTo(oldContext);

    PGX_LOG(RUNTIME, TRACE, "process_computed_results: streaming returned %s", result ? "true" : "false");
    return result;
}

extern "C" auto add_tuple_to_result(const int64_t value) -> bool {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, TRACE, "add_tuple_to_result: called with value=%ld", value);
    PGX_LOG(RUNTIME, TRACE, "add_tuple_to_result: numComputedColumns=%d", g_computed_results.numComputedColumns);

    if (!validate_memory_context_safety("add_tuple_to_result")) {
        return false;
    }

    if (g_computed_results.numComputedColumns > 0) {
        PGX_LOG(RUNTIME, DEBUG, "add_tuple_to_result: Streaming computed results");
        return process_computed_results_for_streaming();
    }

    PGX_LOG(RUNTIME, TRACE, "add_tuple_to_result: No computed results available");
    return false;
}

//==============================================================================
// Field Access Functions
//==============================================================================

extern "C" bool get_bool_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    return pgx_lower::runtime::extract_field<bool>(field_index, is_null);
}

extern "C" int32_t get_field_type_oid(int32_t field_index) {
    // Get the PostgreSQL type OID for a field using the global tuple descriptor
    if (!g_current_tuple_passthrough.tupleDesc || 
        field_index < 0 || 
        field_index >= g_current_tuple_passthrough.tupleDesc->natts) {
        PGX_INFO_C("g_current_tuple_passthrough seems to be invalid");
        return 0;  // Invalid OID
    }
    
    return TupleDescAttr(g_current_tuple_passthrough.tupleDesc, field_index)->atttypid;
}

extern "C" const char* get_string_field(void* tuple_handle, int32_t field_index, bool* is_null, int32_t* length, int32_t type_oid) {
    PGX_IO(RUNTIME);
#ifdef POSTGRESQL_EXTENSION
    if (!pgx_lower::runtime::check_memory_context_safety()) {
        PGX_ERROR("get_string_field: Memory context unsafe for PostgreSQL operations");
        *is_null = true;
        *length = 0;
        return nullptr;
    }

    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        *length = 0;
        return nullptr;
    }

    const int attr_num = field_index + 1;
    if (attr_num > g_current_tuple_passthrough.tupleDesc->natts) {
        *is_null = true;
        *length = 0;
        return nullptr;
    }

    bool isnull;
    Datum value = heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num, 
                              g_current_tuple_passthrough.tupleDesc, &isnull);
    *is_null = isnull;
    if (isnull) {
        *length = 0;
        return nullptr;
    }

    // Use the type OID passed from caller to determine how to extract the string
    // This avoids redundant TupleDesc access since caller already has the type OID
    switch (type_oid) {
    case TEXTOID: {
        text* pg_text = DatumGetTextPP(value);
        *length = VARSIZE_ANY_EXHDR(pg_text);
        return VARDATA_ANY(pg_text);
    }
    case VARCHAROID: {
        VarChar* pg_varchar = DatumGetVarCharPP(value);
        *length = VARSIZE_ANY_EXHDR(pg_varchar);
        return VARDATA_ANY(pg_varchar);
    }
    case BPCHAROID: {
        BpChar* pg_bpchar = DatumGetBpCharPP(value);
        *length = VARSIZE_ANY_EXHDR(pg_bpchar);
        return VARDATA_ANY(pg_bpchar);
    }
    case NAMEOID: {
        // Handle name type (fixed 64-byte)
        Name name = DatumGetName(value);
        *length = strlen(NameStr(*name));
        return NameStr(*name);
    }
    default:
        // Unknown type - try to convert to string
        PGX_WARNING("get_string_field: Unexpected type OID %u for string field", type_oid);
        *is_null = true;
        *length = 0;
        return nullptr;
    }
#else
    // Non-PostgreSQL build - return empty
    *is_null = true;
    *length = 0;
    return nullptr;
#endif
}

extern "C" int64_t get_text_field(void* tuple_handle, const int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    if (!g_current_tuple_passthrough.originalTuple || !g_current_tuple_passthrough.tupleDesc) {
        *is_null = true;
        return 0;
    }

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

//==============================================================================
// Result Storage Functions
//==============================================================================

extern "C" void prepare_computed_results(int32_t numColumns) {
    PGX_IO(RUNTIME);
    g_computed_results.resize(numColumns);
}

extern "C" void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME,
            IO,
            "store_bigint_result IN: columnIndex=%d, value=%ld, isNull=%s",
            columnIndex,
            value,
            isNull ? "true" : "false");
    Datum datum = Int64GetDatum(value);
    g_computed_results.setResult(columnIndex, datum, isNull, INT8OID);
    PGX_LOG(RUNTIME, IO, "store_bigint_result OUT");
}

extern "C" void mark_results_ready_for_streaming() {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, IO, "mark_results_ready_for_streaming IN");
    g_jit_results_ready = true;
    PGX_LOG(RUNTIME, DEBUG, "AFTER: g_jit_results_ready = %d", g_jit_results_ready);

    PGX_LOG(RUNTIME, TRACE, "Validation: g_computed_results.numComputedColumns = %d", g_computed_results.numComputedColumns);
    if (g_computed_results.numComputedColumns > 0) {
        PGX_LOG(RUNTIME,
                TRACE,
                "Validation: First computed value = %ld",
                DatumGetInt64(g_computed_results.computedValues[0]));
    }
    PGX_LOG(RUNTIME, IO, "mark_results_ready_for_streaming OUT");
}

//==============================================================================
// Template-Based Field Extraction Functions
//==============================================================================

extern "C" double get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<double>(field_index, is_null);
}

extern "C" int16_t get_int16_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<int16_t>(field_index, is_null);
}

extern "C" int32_t get_int32_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<int32_t>(field_index, is_null);
}

extern "C" int64_t get_int64_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<int64_t>(field_index, is_null);
}

extern "C" float get_float32_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<float>(field_index, is_null);
}

extern "C" double get_float64_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<double>(field_index, is_null);
}

extern "C" int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    PGX_IO(RUNTIME);
    return pgx_lower::runtime::extract_field<int32_t>(field_index, is_null);
}

//==============================================================================
// Runtime Database Interface
//==============================================================================

extern "C" void* DataSource_get(runtime::VarLen32 description) {
    PGX_IO(RUNTIME);
    try {
        return pgx_lower::compiler::runtime::PostgreSQLDataSource::createFromDescription(description);
    } catch (const std::exception& e) {
        PGX_ERROR("DataSource_get: Failed to create PostgreSQL data source: %s", e.what());
        return nullptr;
    } catch (...) {
        PGX_ERROR("DataSource_get: Unknown exception while creating PostgreSQL data source");
        return nullptr;
    }
}

//==============================================================================
// MLIR Wrapper Functions
//==============================================================================

extern "C" int32_t get_int32_field_mlir(int64_t iteration_signal, int32_t field_index) {
    PGX_IO(RUNTIME);
    bool is_null;
    return pgx_lower::runtime::extract_field<int32_t>(field_index, &is_null);
}

extern "C" int64_t get_int64_field_mlir(int64_t iteration_signal, int32_t field_index) {
    PGX_IO(RUNTIME);
    bool is_null;
    return pgx_lower::runtime::extract_field<int64_t>(field_index, &is_null);
}

extern "C" float get_float32_field_mlir(int64_t iteration_signal, int32_t field_index) {
    PGX_IO(RUNTIME);
    bool is_null;
    return pgx_lower::runtime::extract_field<float>(field_index, &is_null);
}

extern "C" double get_float64_field_mlir(int64_t iteration_signal, int32_t field_index) {
    PGX_IO(RUNTIME);
    bool is_null;
    return pgx_lower::runtime::extract_field<double>(field_index, &is_null);
}