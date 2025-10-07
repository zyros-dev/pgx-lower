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
    static void* beginDatum(
        uint32_t type_oid,
        uint32_t sort_op_oid,
        uint32_t collation_oid,
        bool nulls_first,
        uint32_t work_mem_kb
    );

    static void putDatum(
        void* sort_state,
        uint64_t value,
        bool is_null
    );

    static void performSort(void* sort_state);

    static bool getDatum(
        void* sort_state,
        void* value_ptr,
        void* is_null_ptr
    );

    static void end(void* sort_state);
};

} // namespace runtime

#endif // RUNTIME_PG_SORT_RUNTIME_H
