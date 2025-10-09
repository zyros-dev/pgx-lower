#ifndef RUNTIME_PG_SORT_RUNTIME_H
#define RUNTIME_PG_SORT_RUNTIME_H

#include "lingodb/runtime/helpers.h"

namespace runtime {

/**
 * PostgreSQL tuplesort-based sorting runtime.
 *
 * Thin wrapper around PostgreSQL's tuplesort API for datum-based sorting.
 * Handles memory management, disk spilling, and type-aware comparisons.
 */
class PgSortRuntime {
   public:
    static void*
    beginDatum(uint32_t type_oid, uint32_t sort_op_oid, uint32_t collation_oid, bool nulls_first, uint32_t work_mem_kb);

    static void putDatum(void* sort_state, uint64_t value, bool is_null);

    static void performSort(void* sort_state);

    static bool getDatum(void* sort_state, void* value_ptr, void* is_null_ptr);

    static void end(void* sort_state);

    static void* beginHeapSort(const uint32_t* all_type_oids, const int32_t* all_typmods, int32_t num_total_cols,
                               const int32_t* sort_key_indices, const uint32_t* sort_op_oids,
                               const int32_t* sort_directions, int32_t num_sort_keys);

    static void putHeapTuple(void* sort_state,
                             const uintptr_t* datums, // Array of Datum values
                             const bool* nulls, // Array of null flags
                             int32_t num_cols);

    static bool getHeapTuple(void* sort_state,
                             uintptr_t* datums_out, // OUT: Array to fill with Datums
                             bool* nulls_out, // OUT: Array to fill with null flags
                             int32_t num_cols);
};

} // namespace runtime

#endif // RUNTIME_PG_SORT_RUNTIME_H
