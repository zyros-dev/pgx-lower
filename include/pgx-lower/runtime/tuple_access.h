#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/execution/postgres/my_executor.h"
// Removed mlir_runner.h include to avoid DestReceiver conflicts
#include "pgx-lower/frontend/SQL/query_analyzer.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#include "executor/executor.h"

#include <vector>

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "access/htup_details.h"
#include "utils/lsyscache.h"
#include <tcop/dest.h>
#include "access/heapam.h"

#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "postgres.h"
#include "tcop/dest.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/builtins.h"
#include "access/heapam.h"
}
#endif

// Global context for tuple scanning - used by external function
struct TupleScanContext {
    TableScanDesc scanDesc{};
    TupleDesc tupleDesc{};
    bool hasMore{};
    int64_t currentValue{};
};

struct ComputedResultStorage {
    std::vector<Datum> computedValues; // Computed expression results
    std::vector<bool> computedNulls; // Null flags for computed results
    std::vector<Oid> computedTypes; // PostgreSQL type OIDs for computed values
    int numComputedColumns = 0; // Number of computed columns in current query

    void clear() {
        computedValues.clear();
        computedNulls.clear();
        computedTypes.clear();
        numComputedColumns = 0;
    }

    void resize(int numColumns) {
        PGX_NOTICE("ComputedResultStorage::resize called with numColumns=" + std::to_string(numColumns));
        numComputedColumns = numColumns;
        computedValues.resize(numColumns, 0);
        computedNulls.resize(numColumns, true);
        computedTypes.resize(numColumns, InvalidOid);
        PGX_NOTICE("ComputedResultStorage::resize completed, numComputedColumns=" + std::to_string(numComputedColumns));
    }

    void setResult(int columnIndex, Datum value, bool isNull, Oid typeOid) {
        if (columnIndex >= 0 && columnIndex < numComputedColumns) {
            computedValues[columnIndex] = value;
            computedNulls[columnIndex] = isNull;
            computedTypes[columnIndex] = typeOid;
        }
    }
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
#ifdef POSTGRESQL_EXTENSION
        if (originalTuple) {
            heap_freetuple(originalTuple);
            originalTuple = nullptr;
        }
#endif
    }

    // Return a simple signal that we have a valid tuple
    // MLIR only needs to know "continue iterating" vs "end of table"
    // (All actual data passes through via originalTuple)
    int64_t getIterationSignal() const { return originalTuple ? 1 : 0; }
};

extern ComputedResultStorage g_computed_results;

// Global to hold field indices for current query (temporary hack)
extern std::vector<int> g_field_indices;

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
        if (!isActive || !dest || !slot) {
            return false;
        }
        
        // Check if we have computed-only results (aggregates) or regular tuple results
        bool hasComputedResults = false;
        for (int col : selectedColumns) {
            if (col == -1) {
                hasComputedResults = true;
                break;
            }
        }
        
        // For computed-only results, we don't need an original tuple
        if (!hasComputedResults && !passthrough.originalTuple) {
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

                    if (origColumnIndex >= 0 && origTupleDesc && origColumnIndex < origTupleDesc->natts) {
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

// Global variables for tuple processing and computed result storage
extern TupleStreamer g_tuple_streamer;
extern PostgreSQLTuplePassthrough g_current_tuple_passthrough;
extern Oid g_jit_table_oid; // Table OID for JIT-managed table access

extern "C" {
// MLIR Interface: Read next tuple for iteration control
// Returns: tuple pointer as int64_t if valid tuple, 0 if end of table
auto read_next_tuple_from_table(void* tableHandle) -> int64_t;

// MLIR Interface: Stream complete PostgreSQL tuple to output
// The 'value' parameter is ignored - it's just MLIR's iteration signal
auto add_tuple_to_result(int64_t value) -> bool;

auto open_postgres_table(const char* tableName) -> void*;

void close_postgres_table(void* tableHandle);

// Legacy interface for simple tuple access (kept for compatibility)
auto get_next_tuple() -> int64_t;

// Typed field access functions for PostgreSQL dialect
auto get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int32_t;
// MLIR-compatible wrapper for JIT integration
extern "C" int32_t get_int_field_mlir(int64_t iteration_signal, int32_t field_index);
auto get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int64_t;
auto get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) -> double;

// Result storage functions for expressions
void store_int_result(int32_t columnIndex, int32_t value, bool isNull);
void store_bool_result(int32_t columnIndex, bool value, bool isNull);
void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
void store_text_result(int32_t columnIndex, const char* value, bool isNull);
void prepare_computed_results(int32_t numColumns);
void mark_results_ready_for_streaming();

// Global flag to indicate results are ready for streaming (defined in tuple_access.cpp)
extern bool g_jit_results_ready;

// Type-aware field extractor that handles all PostgreSQL types
void store_field_as_datum(int32_t columnIndex, int64_t iteration_signal, int32_t field_index);

// Aggregate functions
int64_t sum_aggregate(void* table_handle);

auto get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) -> double;

// Critical runtime function for DB dialect (GetExternalOp lowering)
void* DataSource_get(runtime::VarLen32 description);
}

namespace pgx_lower {
namespace runtime {
// Memory context safety check for PostgreSQL operations
// Returns true if the current memory context is safe for PostgreSQL operations
bool check_memory_context_safety();
} // namespace runtime
} // namespace pgx_lower