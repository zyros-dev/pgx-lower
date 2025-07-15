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