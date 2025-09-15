#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/execution/postgres/my_executor.h"
#include "pgx-lower/frontend/SQL/query_analyzer.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"
#include "executor/executor.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "access/htup_details.h"
#include "utils/lsyscache.h"
#include <tcop/dest.h>
#include "access/heapam.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "tcop/dest.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/builtins.h"
#include "utils/numeric.h"
}
#endif

// Data Structures
struct ColumnMetadata {
    char name[64];
    int32_t type_oid;
    int32_t attnum;
};

struct ComputedResultStorage {
    std::vector<Datum> computedValues;
    std::vector<bool> computedNulls;
    std::vector<Oid> computedTypes;
    int numComputedColumns = 0;

    void clear() {
        computedValues.clear();
        computedNulls.clear();
        computedTypes.clear();
        numComputedColumns = 0;
    }

    void resize(int numColumns) {
        PGX_LOG(RUNTIME, DEBUG, "ComputedResultStorage::resize called with numColumns=%d", numColumns);
        numComputedColumns = numColumns;
        computedValues.resize(numColumns, 0);
        computedNulls.resize(numColumns, true);
        computedTypes.resize(numColumns, InvalidOid);
        PGX_LOG(RUNTIME, DEBUG, "ComputedResultStorage::resize completed, numComputedColumns=%zu", numComputedColumns);
    }

    void setResult(int columnIndex, Datum value, bool isNull, Oid typeOid) {
        if (columnIndex >= 0 && columnIndex < numComputedColumns) {
            computedValues[columnIndex] = value;
            computedNulls[columnIndex] = isNull;
            computedTypes[columnIndex] = typeOid;
        }
    }
};

struct PostgreSQLTuplePassthrough {
    HeapTuple originalTuple;
    TupleDesc tupleDesc;

    PostgreSQLTuplePassthrough()
    : originalTuple(nullptr)
    , tupleDesc(nullptr) {}

    ~PostgreSQLTuplePassthrough() {
#ifdef POSTGRESQL_EXTENSION
        if (originalTuple) {
            heap_freetuple(originalTuple);
            originalTuple = nullptr;
        }
#endif
    }

    int64_t getIterationSignal() const { return originalTuple ? 1 : 0; }
};

extern ComputedResultStorage g_computed_results;

struct TupleStreamer {
    DestReceiver* dest;
    TupleTableSlot* slot;
    bool isActive;
    std::vector<int> selectedColumns;

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

    auto streamCompletePostgreSQLTuple(const PostgreSQLTuplePassthrough& passthrough) const -> bool {
        if (!isActive || !dest || !slot) {
            return false;
        }

        bool hasComputedResults = false;
        for (int col : selectedColumns) {
            if (col == -1) {
                hasComputedResults = true;
                break;
            }
        }

        if (!hasComputedResults && !passthrough.originalTuple) {
            return false;
        }

        try {
            ExecClearTuple(slot);

            const auto origTupleDesc = passthrough.tupleDesc;
            const auto resultTupleDesc = slot->tts_tupleDescriptor;

            for (int i = 0; i < resultTupleDesc->natts; i++) {
                bool isnull = false;

                if (i < selectedColumns.size()) {
                    const int origColumnIndex = selectedColumns[i];

                    if (origColumnIndex >= 0 && origTupleDesc && origColumnIndex < origTupleDesc->natts) {
                        const auto value = heap_getattr(passthrough.originalTuple, origColumnIndex + 1, origTupleDesc,
                                                        &isnull);
                        slot->tts_values[i] = value;
                        slot->tts_isnull[i] = isnull;
                    } else if (origColumnIndex == -1 && i < g_computed_results.numComputedColumns) {
                        slot->tts_values[i] = g_computed_results.computedValues[i];
                        slot->tts_isnull[i] = g_computed_results.computedNulls[i];
                    } else {
                        slot->tts_values[i] = static_cast<Datum>(0);
                        slot->tts_isnull[i] = true;
                    }
                } else {
                    slot->tts_values[i] = static_cast<Datum>(0);
                    slot->tts_isnull[i] = true;
                }
            }

            slot->tts_nvalid = resultTupleDesc->natts;
            ExecStoreVirtualTuple(slot);

            return dest->receiveSlot(slot, dest);
        } catch (...) {
            PGX_WARNING("Exception caught in streamCompletePostgreSQLTuple");
            return false;
        }
    }

    void shutdown() {
        isActive = false;
        dest = nullptr;
        slot = nullptr;
    }
};

// Global Variables
extern std::vector<int> g_field_indices;
extern TupleStreamer g_tuple_streamer;
extern PostgreSQLTuplePassthrough g_current_tuple_passthrough;
extern Oid g_jit_table_oid;
extern bool g_jit_results_ready;

// C Interface - Table Management
extern "C" {
auto read_next_tuple_from_table(void* tableHandle) -> int64_t;
auto add_tuple_to_result(int64_t value) -> bool;
auto open_postgres_table(const char* tableName) -> void*;
void close_postgres_table(void* tableHandle);

// C Interface - Field Access
auto get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int32_t;
auto get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int64_t;
auto get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) -> Numeric;

// C Interface - Result Storage
void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
void prepare_computed_results(int32_t numColumns);
void mark_results_ready_for_streaming();

// C Interface - Runtime Functions
void* DataSource_get(runtime::VarLen32 description);
} // extern "C"

// C++ Interface
namespace pgx_lower { namespace runtime {
bool check_memory_context_safety();
}} // namespace pgx_lower::runtime