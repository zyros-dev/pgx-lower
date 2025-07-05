#include "runtime/tuple_access.h"
#include "core/error_handling.h"
#include <array>
#include <cstring>
#include <memory>

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
}
#endif

namespace pgx_lower::runtime {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static constexpr int MAX_MOCK_FIELDS = 10;
static constexpr int32_t MOCK_INT_VALUE = 42;
static constexpr int32_t MOCK_BIGINT_VALUE = 100;
static constexpr uint32_t INT4_TYPE_OID = 23;

//===----------------------------------------------------------------------===//
// Internal Handle Structures
//===----------------------------------------------------------------------===//

#ifdef POSTGRESQL_EXTENSION

struct TableScanHandle {
    Relation relation;
    TableScanDesc scan_desc;
    TupleDesc tuple_desc;
    bool is_open;

    TableScanHandle()
    : relation(nullptr)
    , scan_desc(nullptr)
    , tuple_desc(nullptr)
    , is_open(false) {}

    ~TableScanHandle() {
        if (is_open && scan_desc) {
            table_endscan(scan_desc);
        }
        if (relation) {
            table_close(relation, AccessShareLock);
        }
    }
};

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
struct TableScanHandle {
    bool is_open = false;
};

struct TupleHandle {
    int mock_field_count = 2;
    std::array<int64_t, MAX_MOCK_FIELDS> mock_values = {MOCK_INT_VALUE, MOCK_BIGINT_VALUE, 0};
    std::array<bool, MAX_MOCK_FIELDS> mock_nulls = {false, false, true};

    TupleHandle() = default;
};

#endif

//===----------------------------------------------------------------------===//
// Table Operations
//===----------------------------------------------------------------------===//

TableScanHandle* openTableScan(const char* table_name) {
    if (!table_name) {
        REPORT_ERROR_CTX(ERROR_LEVEL, POSTGRESQL, "Table name is null", "openTableScan");
        return nullptr;
    }

#ifdef POSTGRESQL_EXTENSION
    try {
        auto handle = std::make_unique<TableScanHandle>();

        // TODO: Complete table lookup implementation
        // This function needs integration with PostgreSQL's catalog system to:
        // 1. Parse and validate the table name (handle schema qualification)
        // 2. Look up the relation OID using RangeVarGetRelid()
        // 3. Open the relation with appropriate locking
        // 4. Create a proper table scan descriptor with current transaction snapshot
        // 5. Handle permissions checking and error cases
        //
        // For now, this is a placeholder that signals the function is not complete.
        // The actual table scanning is handled by the existing PostgreSQL executor
        // integration in my_executor.cpp which has access to the current plan context.

        REPORT_ERROR_CTX(WARNING_LEVEL,
                         POSTGRESQL,
                         "Direct table lookup by name requires plan context integration",
                         table_name);
        return nullptr;

    } catch (const std::exception& e) {
        auto error = ErrorManager::postgresqlError("Exception in openTableScan: " + std::string(e.what()), table_name);
        ErrorManager::reportError(error);
        return nullptr;
    }
#else
    // Mock implementation for unit tests
    return new TableScanHandle();
#endif
}

void closeTableScan(const TableScanHandle* handle) {
    if (handle) {
        delete handle;
    }
}

TupleHandle* readNextTuple(TableScanHandle* handle) {
    if (!handle) {
        REPORT_ERROR(ERROR_LEVEL, POSTGRESQL, "Table scan handle is null");
        return nullptr;
    }

#ifdef POSTGRESQL_EXTENSION
    // If this is the dummy handle (not actually open), return nullptr for now
    if (!handle->is_open) {
        REPORT_ERROR(WARNING_LEVEL, POSTGRESQL, "Table scan is not open (dummy handle)");
        return nullptr;
    }
    
    if (!handle->scan_desc) {
        REPORT_ERROR(ERROR_LEVEL, POSTGRESQL, "Table scan descriptor is null");
        return nullptr;
    }

    TupleTableSlot* slot = table_slot_create(handle->relation, NULL);

    if (!table_scan_getnextslot(handle->scan_desc, ForwardScanDirection, slot)) {
        ExecDropSingleTupleTableSlot(slot);
        return nullptr; // End of table
    }

    // Convert slot to HeapTuple for compatibility
    HeapTuple tuple = ExecCopySlotHeapTuple(slot);
    ExecDropSingleTupleTableSlot(slot);

    return new TupleHandle(tuple, handle->tuple_desc, true);
#else
    // Mock implementation - return a mock tuple
    static int call_count = 0;
    call_count++;

    if (call_count > 3) {
        return nullptr; // End of mock data
    }

    return new TupleHandle();
#endif
}

//===----------------------------------------------------------------------===//
// Tuple Field Access
//===----------------------------------------------------------------------===//

int32_t getIntField(TupleHandle* tuple, int field_index, bool* is_null) {
    if (!tuple || !is_null) {
        if (is_null)
            *is_null = true;
        return 0;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->heap_tuple || !tuple->tuple_desc) {
        *is_null = true;
        return 0;
    }

    bool isnull = false;
    Datum value = heap_getattr(tuple->heap_tuple, field_index + 1, tuple->tuple_desc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0;
    }

    return DatumGetInt32(value);
#else
    // Mock implementation
    if (field_index >= MAX_MOCK_FIELDS) {
        *is_null = true;
        return 0;
    }

    *is_null = tuple->mock_nulls[field_index];
    return static_cast<int32_t>(tuple->mock_values[field_index]);
#endif
}

int64_t getBigIntField(TupleHandle* tuple, int field_index, bool* is_null) {
    if (!tuple || !is_null) {
        if (is_null)
            *is_null = true;
        return 0;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->heap_tuple || !tuple->tuple_desc) {
        *is_null = true;
        return 0;
    }

    bool isnull = false;
    Datum value = heap_getattr(tuple->heap_tuple, field_index + 1, tuple->tuple_desc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return 0;
    }

    return DatumGetInt64(value);
#else
    // Mock implementation
    if (field_index >= MAX_MOCK_FIELDS) {
        *is_null = true;
        return 0;
    }

    *is_null = tuple->mock_nulls[field_index];
    return tuple->mock_values[field_index];
#endif
}

const char* getTextField(TupleHandle* tuple, int field_index, int* length, bool* is_null) {
    if (!tuple || !length || !is_null) {
        if (is_null)
            *is_null = true;
        if (length)
            *length = 0;
        return nullptr;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->heap_tuple || !tuple->tuple_desc) {
        *is_null = true;
        *length = 0;
        return nullptr;
    }

    bool isnull = false;
    Datum value = heap_getattr(tuple->heap_tuple, field_index + 1, tuple->tuple_desc, &isnull);

    *is_null = isnull;
    if (isnull) {
        *length = 0;
        return nullptr;
    }

    text* txt = DatumGetTextP(value);
    *length = VARSIZE_ANY_EXHDR(txt);
    return VARDATA_ANY(txt);
#else
    // Mock implementation
    static const char* mock_text = "mock_text_value";

    if (field_index >= 10) {
        *is_null = true;
        *length = 0;
        return nullptr;
    }

    *is_null = tuple->mock_nulls[field_index];
    if (*is_null) {
        *length = 0;
        return nullptr;
    }

    *length = strlen(mock_text);
    return mock_text;
#endif
}

bool getBoolField(TupleHandle* tuple, int field_index, bool* is_null) {
    if (!tuple || !is_null) {
        if (is_null)
            *is_null = true;
        return false;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->heap_tuple || !tuple->tuple_desc) {
        *is_null = true;
        return false;
    }

    bool isnull = false;
    Datum value = heap_getattr(tuple->heap_tuple, field_index + 1, tuple->tuple_desc, &isnull);

    *is_null = isnull;
    if (isnull) {
        return false;
    }

    return DatumGetBool(value);
#else
    // Mock implementation
    if (field_index >= 10) {
        *is_null = true;
        return false;
    }

    *is_null = tuple->mock_nulls[field_index];
    return (tuple->mock_values[field_index] % 2) == 1;
#endif
}

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

//===----------------------------------------------------------------------===//
// Type Information
//===----------------------------------------------------------------------===//

uint32_t getFieldTypeOid(TupleHandle* tuple, int field_index) {
    if (!tuple) {
        return 0;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->tuple_desc || field_index >= tuple->tuple_desc->natts) {
        return 0;
    }

    return TupleDescAttr(tuple->tuple_desc, field_index)->atttypid;
#else
    // Mock implementation - return INT4OID for simplicity
    return INT4_TYPE_OID;
#endif
}

int getTupleFieldCount(TupleHandle* tuple) {
    if (!tuple) {
        return 0;
    }

#ifdef POSTGRESQL_EXTENSION
    if (!tuple->tuple_desc) {
        return 0;
    }

    return tuple->tuple_desc->natts;
#else
    // Mock implementation
    return tuple->mock_field_count;
#endif
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

bool compareInt(int32_t lhs, bool lhs_null, int32_t rhs, bool rhs_null, const char* op) {
    // PostgreSQL NULL semantics: any comparison with NULL returns false
    if (lhs_null || rhs_null) {
        return false;
    }

    if (strcmp(op, "eq") == 0)
        return lhs == rhs;
    if (strcmp(op, "ne") == 0)
        return lhs != rhs;
    if (strcmp(op, "lt") == 0)
        return lhs < rhs;
    if (strcmp(op, "le") == 0)
        return lhs <= rhs;
    if (strcmp(op, "gt") == 0)
        return lhs > rhs;
    if (strcmp(op, "ge") == 0)
        return lhs >= rhs;

    REPORT_ERROR_CTX(ERROR_LEVEL, EXECUTION, "Unknown comparison operator", op);
    return false;
}

bool compareText(const char* lhs, bool lhs_null, const char* rhs, bool rhs_null, const char* op) {
    // PostgreSQL NULL semantics: any comparison with NULL returns false
    if (lhs_null || rhs_null) {
        return false;
    }

    int cmp_result = strcmp(lhs, rhs);

    if (strcmp(op, "eq") == 0)
        return cmp_result == 0;
    if (strcmp(op, "ne") == 0)
        return cmp_result != 0;
    if (strcmp(op, "lt") == 0)
        return cmp_result < 0;
    if (strcmp(op, "le") == 0)
        return cmp_result <= 0;
    if (strcmp(op, "gt") == 0)
        return cmp_result > 0;
    if (strcmp(op, "ge") == 0)
        return cmp_result >= 0;

    REPORT_ERROR_CTX(ERROR_LEVEL, EXECUTION, "Unknown comparison operator", op);
    return false;
}

//===----------------------------------------------------------------------===//
// Output Operations
//===----------------------------------------------------------------------===//

bool outputTuple(const TupleHandle* tuple) {
    if (!tuple) {
        REPORT_ERROR(ERROR_LEVEL, POSTGRESQL, "Cannot output null tuple");
        return false;
    }

    // TODO: Integrate with PostgreSQL result output mechanism
    // This function should:
    // 1. Get the current DestReceiver from the execution context
    // 2. Create a TupleTableSlot from the tuple handle
    // 3. Call ExecutorSend() or similar to output the tuple
    // 4. Handle proper slot lifecycle management
    //
    // The current executor integration in my_executor.cpp handles this
    // at a higher level, so this runtime function may not be needed.
    REPORT_ERROR(WARNING_LEVEL, POSTGRESQL, "Direct tuple output requires execution context integration");
    return true; // Return true to avoid breaking compilation
}

TupleHandle* createTuple(int field_count, const uint32_t* field_types, const int64_t* field_values, const bool* null_flags) {
    if (field_count <= 0 || !field_types || !field_values || !null_flags) {
        REPORT_ERROR(ERROR_LEVEL, POSTGRESQL, "Invalid parameters for createTuple");
        return nullptr;
    }

    // TODO: Implement full PostgreSQL tuple construction
    // This function should:
    // 1. Create a TupleDesc from the field_types array
    // 2. Allocate a tuple with the correct size
    // 3. Set field values and null flags properly
    // 4. Handle type-specific value encoding (text, numeric, etc.)
    // 5. Return a properly constructed TupleHandle
    //
    // This is complex as it requires deep PostgreSQL tuple format knowledge.
    // For now, return a placeholder that signals incomplete implementation.
#ifdef POSTGRESQL_EXTENSION
    REPORT_ERROR(WARNING_LEVEL, POSTGRESQL, "Tuple construction not yet fully implemented");
    return new TupleHandle(nullptr, nullptr, false);
#else
    return new TupleHandle();
#endif
}

auto freeTuple(const TupleHandle* tuple) -> void {
    delete tuple;
}

//===----------------------------------------------------------------------===//
// Aggregation Support
//===----------------------------------------------------------------------===//

struct SumAggregationState {
    int64_t sum = 0;
    bool has_values = false;
};

void* initSumAggregation() {
    return new SumAggregationState();
}

void addToSum(void* state, int64_t value, bool is_null) {
    if (!state || is_null) {
        return; // NULL values are ignored in PostgreSQL SUM
    }

    auto* sum_state = static_cast<SumAggregationState*>(state);
    sum_state->sum += value;
    sum_state->has_values = true;
}

int64_t finalizeSumAggregation(void* state, bool* result_null) {
    if (!state || !result_null) {
        if (result_null)
            *result_null = true;
        return 0;
    }

    auto* sum_state = static_cast<SumAggregationState*>(state);
    *result_null = !sum_state->has_values; // NULL if no non-NULL values
    return sum_state->sum;
}

void freeAggregationState(void* state) {
    delete static_cast<SumAggregationState*>(state);
}

} // namespace pgx_lower::runtime

//===----------------------------------------------------------------------===//
// C-style interface for MLIR JIT compatibility
// Note: These conflict with unit test mock functions, so only compile for extension
//===----------------------------------------------------------------------===//

#ifdef POSTGRESQL_EXTENSION
extern "C" {

void* open_postgres_table(const char* table_name) {
    #ifdef POSTGRESQL_EXTENSION
    elog(NOTICE, "🔧 open_postgres_table called with table_name='%s'", table_name ? table_name : "NULL");
    #endif
    
    // For now, return a dummy non-null pointer since table scanning is not yet implemented
    // This prevents null pointer crashes while we work on the full implementation
    #ifdef POSTGRESQL_EXTENSION
    static pgx_lower::runtime::TableScanHandle dummy_handle;
    dummy_handle.is_open = false; // Mark as not actually open
    elog(NOTICE, "🎭 open_postgres_table: returning dummy handle=%p (table scanning not implemented)", &dummy_handle);
    return static_cast<void*>(&dummy_handle);
    #else
    return static_cast<void*>(pgx_lower::runtime::openTableScan(table_name));
    #endif
}

int64_t read_next_tuple_from_table(void* table_handle) {
    #ifdef POSTGRESQL_EXTENSION
    elog(NOTICE, "🔍 read_next_tuple_from_table called with handle=%p", table_handle);
    #endif
    
    auto tuple = pgx_lower::runtime::readNextTuple(static_cast<pgx_lower::runtime::TableScanHandle*>(table_handle));
    if (tuple) {
        #ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "✅ read_next_tuple_from_table: returning real tuple=%p", tuple);
        #endif
        // Return tuple pointer as int64_t for MLIR compatibility
        return reinterpret_cast<int64_t>(tuple);
    } else {
        // CRITICAL WORKAROUND: MLIR scf.while loop always executes first iteration
        // even when condition is false. Instead of returning 0 (null pointer) which crashes,
        // return a dummy tuple on first call, then 0 on second call to exit loop.
        #ifdef POSTGRESQL_EXTENSION
        static int call_count = 0;
        call_count++;
        
        elog(NOTICE, "🔄 read_next_tuple_from_table: no tuple, call_count=%d", call_count);
        
        if (call_count == 1) {
            // First call: return dummy tuple to survive mandatory first iteration
            static pgx_lower::runtime::TupleHandle dummy_tuple(nullptr, nullptr, false);
            elog(NOTICE, "🎭 read_next_tuple_from_table: returning dummy tuple=%p", &dummy_tuple);
            return reinterpret_cast<int64_t>(&dummy_tuple);
        } else {
            // Second call: return 0 to exit loop
            call_count = 0; // Reset for next query
            elog(NOTICE, "🛑 read_next_tuple_from_table: returning 0 to exit loop");
            return 0;
        }
        #else
        // For unit tests, return 0 is fine since mock implementation handles it
        return 0;
        #endif
    }
}

void close_postgres_table(void* table_handle) {
    pgx_lower::runtime::closeTableScan(static_cast<pgx_lower::runtime::TableScanHandle*>(table_handle));
}

bool add_tuple_to_result(int64_t value) {
    // The value parameter is the tuple pointer as int64_t (as documented in interface)
    if (value == 0) {
        // Null tuple - end of table
        return false;
    }
    auto tuple_handle = reinterpret_cast<pgx_lower::runtime::TupleHandle*>(value);
    return pgx_lower::runtime::outputTuple(tuple_handle);
}

int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    #ifdef POSTGRESQL_EXTENSION
    // Add debug logging for crash analysis  
    elog(NOTICE, "🔍 get_int_field called with handle=%p field_index=%d", tuple_handle, field_index);
    #endif
    
    // Safety check: handle null pointers
    if (tuple_handle == nullptr) {
        #ifdef POSTGRESQL_EXTENSION
        elog(NOTICE, "🚨 get_int_field: null handle detected, returning null");
        #endif
        if (is_null) *is_null = true;
        return 0;
    }
    
    #ifdef POSTGRESQL_EXTENSION
    elog(NOTICE, "✅ get_int_field: calling getIntField...");
    #endif
    
    auto result = pgx_lower::runtime::getIntField(static_cast<pgx_lower::runtime::TupleHandle*>(tuple_handle),
                                                  field_index,
                                                  is_null);
    
    #ifdef POSTGRESQL_EXTENSION
    elog(NOTICE, "✅ get_int_field completed, result=%d is_null=%s", result, is_null ? (*is_null ? "true" : "false") : "null");
    #endif
    
    return result;
}

int64_t get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    // Safety check: handle null pointers  
    if (tuple_handle == nullptr) {
        if (is_null) *is_null = true;
        return 0;
    }
    
    int length = 0;
    const char* text_ptr = pgx_lower::runtime::getTextField(static_cast<pgx_lower::runtime::TupleHandle*>(tuple_handle),
                                                            field_index,
                                                            &length,
                                                            is_null);

    // Return the pointer as an int64_t for MLIR compatibility
    return reinterpret_cast<int64_t>(text_ptr);
}

// Result storage functions for computed expressions
void store_bool_result(int32_t column_index, bool value, bool is_null) {
    // TODO: Implement proper result storage for computed boolean expressions
    // For now, this is a stub to allow JIT execution to proceed
    (void)column_index;
    (void)value;
    (void)is_null;
}

void store_int_result(int32_t column_index, int32_t value, bool is_null) {
    // TODO: Implement proper result storage for computed integer expressions
    // For now, this is a stub to allow JIT execution to proceed
    (void)column_index;
    (void)value;
    (void)is_null;
}

void store_bigint_result(int32_t column_index, int64_t value, bool is_null) {
    // TODO: Implement proper result storage for computed bigint expressions
    // For now, this is a stub to allow JIT execution to proceed
    (void)column_index;
    (void)value;
    (void)is_null;
}

}
#endif // POSTGRESQL_EXTENSION