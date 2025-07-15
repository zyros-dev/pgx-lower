#include "runtime/tuple_access.h"
#include "core/error_handling.h"
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
#include "utils/elog.h"
}
#endif

namespace pgx_lower::runtime {

static constexpr int MAX_MOCK_FIELDS = 10;
static constexpr int32_t MOCK_INT_VALUE = 42;
static constexpr int32_t MOCK_BIGINT_VALUE = 100;
static constexpr uint32_t INT4_TYPE_OID = 23;

#ifdef POSTGRESQL_EXTENSION

struct TupleHandle {
    HeapTuple heap_tuple;
    TupleDesc tuple_desc;
    bool owns_tuple;

    TupleHandle(HeapTuple tuple, TupleDesc desc, bool owns = false)
    : heap_tuple(tuple)
    , tuple_desc(desc)
    , owns_tuple(owns) {}

    ~TupleHandle() {
        if (owns_tuple && heap_tuple) {
            heap_freetuple(heap_tuple);
        }
    }
};

#else

// Mock implementations for unit tests
struct TupleHandle {
    int mock_field_count = 2;
    std::array<int64_t, MAX_MOCK_FIELDS> mock_values = {MOCK_INT_VALUE, MOCK_BIGINT_VALUE, 0};
    std::array<bool, MAX_MOCK_FIELDS> mock_nulls = {false, false, true};

    TupleHandle() = default;
};

#endif

double getNumericField(TupleHandle* tuple, int field_index, bool* is_null) {
    if (!tuple || !is_null) {
        if (is_null)
            *is_null = true;
        return 0.0;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->heap_tuple || !tuple->tuple_desc) {
        *is_null = true;
        return 0.0;
    }

    bool isnull = false;
    Datum value = heap_getattr(tuple->heap_tuple, field_index + 1, tuple->tuple_desc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0.0;
    }

    // Convert numeric to double - this is a simplification
    Datum float8_value = DirectFunctionCall1(numeric_float8, value);
    return DatumGetFloat8(float8_value);
#else
    // Mock implementation
    if (field_index >= 10) {
        *is_null = true;
        return 0.0;
    }

    *is_null = tuple->mock_nulls[field_index];
    return static_cast<double>(tuple->mock_values[field_index]);
#endif
}
} // namespace pgx_lower::runtime

struct PostgreSQLTableHandle {
    Relation rel;
    TableScanDesc scanDesc;
    TupleDesc tupleDesc;
    bool isOpen;
};

extern "C" void* open_postgres_table(const char* tableName) {
#ifdef POSTGRESQL_EXTENSION
    elog(NOTICE, "open_postgres_table called with tableName: %s", tableName ? tableName : "NULL");

    try {
        if (!g_scan_context) {
            elog(NOTICE, "open_postgres_table: g_scan_context is null");
            return nullptr;
        }

        elog(NOTICE, "open_postgres_table: Creating PostgreSQLTableHandle...");
        auto* handle = new PostgreSQLTableHandle();
        // Use the existing scan descriptor from the global context
        handle->scanDesc = g_scan_context->scanDesc;
        handle->tupleDesc = g_scan_context->tupleDesc;
        handle->rel = nullptr;
        handle->isOpen = true;

        elog(NOTICE, "open_postgres_table: Calling heap_rescan...");
        // IMPORTANT: Reset scan to beginning to ensure we read all tuples
        // PostgreSQL 17.5 heap_rescan signature: heap_rescan(scan, key, set_params, allow_strat, allow_sync, allow_pagemode)
        heap_rescan(handle->scanDesc, nullptr, false, false, false, false);

        elog(NOTICE, "open_postgres_table: Successfully created handle, returning %p", handle);
        return handle;
    } catch (...) {
        elog(NOTICE, "open_postgres_table: Exception caught, returning null");
        return nullptr;
    }
#else
    return nullptr;
#endif
}

// MLIR Interface: Read next tuple for iteration control
// Returns: 1 = "we have a tuple", -2 = "end of table"
// Side effect: Preserves COMPLETE PostgreSQL tuple for later streaming
// Architecture: MLIR just iterates, PostgreSQL handles all data types
extern "C" int64_t read_next_tuple_from_table(void* tableHandle) {
#ifdef POSTGRESQL_EXTENSION
    if (!tableHandle) {
        elog(NOTICE, "read_next_tuple_from_table: tableHandle is null");
        return -1;
    }

    const auto* handle = static_cast<PostgreSQLTableHandle*>(tableHandle);
    if (!handle->isOpen || !handle->scanDesc) {
        elog(NOTICE, "read_next_tuple_from_table: handle not open or scanDesc is null");
        return -1;
    }

    elog(NOTICE, "read_next_tuple_from_table: Calling heap_getnext...");
    const auto tuple = heap_getnext(handle->scanDesc, ForwardScanDirection);
    if (tuple == nullptr) {
        elog(NOTICE, "read_next_tuple_from_table: heap_getnext returned null - end of table");
        // End of table reached - return 0 to terminate MLIR loop
        return 0;
    }

    elog(NOTICE, "read_next_tuple_from_table: Found tuple, processing...");

    // Clean up previous tuple if it exists
    if (g_current_tuple_passthrough.originalTuple) {
        heap_freetuple(g_current_tuple_passthrough.originalTuple);
        g_current_tuple_passthrough.originalTuple = nullptr;
    }

    // Preserve the COMPLETE tuple (all columns, all types) for output
    g_current_tuple_passthrough.originalTuple = heap_copytuple(tuple);
    g_current_tuple_passthrough.tupleDesc = handle->tupleDesc;

    elog(NOTICE, "read_next_tuple_from_table: Tuple processed successfully");

    // Return signal: "we have a tuple" (MLIR only uses this for iteration control)
    return g_current_tuple_passthrough.getIterationSignal();
#else
    return 0;
#endif
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
    case BOOLOID: return DatumGetBool(value) ? 1 : 0; // Convert bool to int32
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
        datum = CStringGetDatum(value);
    }
    g_computed_results.setResult(columnIndex, datum, isNull, TEXTOID);
}

extern "C" void prepare_computed_results(int32_t numColumns) {
    g_computed_results.resize(numColumns);
}

//===----------------------------------------------------------------------===//
// C-style interface for MLIR JIT compatibility
//===----------------------------------------------------------------------===//

// get_numeric_field needs to be available for both unit tests and extension
extern "C" double get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) {
#ifdef POSTGRESQL_EXTENSION
    elog(NOTICE, "get_numeric_field called with handle=%p field_index=%d", tuple_handle, field_index);

    // Safety check: handle null pointers
    if (tuple_handle == nullptr) {
        elog(NOTICE, "get_numeric_field: null handle detected, returning null");
        if (is_null)
            *is_null = true;
        return 0.0;
    }

    elog(NOTICE, "get_numeric_field: calling getNumericField...");

    const auto result = pgx_lower::runtime::getNumericField(static_cast<pgx_lower::runtime::TupleHandle*>(tuple_handle),
                                                            field_index,
                                                            is_null);

    elog(NOTICE,
         "get_numeric_field completed, result=%f is_null=%s",
         result,
         is_null ? (*is_null ? "true" : "false") : "null");

    return result;
#else
    return 0.0;
#endif
}

// Note: All other C interface functions are implemented in my_executor.cpp
// Only get_numeric_field is kept here to support both unit tests and extension builds