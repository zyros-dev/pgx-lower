#include "lingodb/runtime/PgSortRuntime.h"

extern "C" {
#include "postgres.h"
#include "utils/tuplesort.h"
#include "utils/datum.h"
}

namespace runtime {

void* PgSortRuntime::beginDatum(const uint32_t type_oid, const uint32_t sort_op_oid, const uint32_t collation_oid,
                                const bool nulls_first, const uint32_t work_mem_kb) {
    return tuplesort_begin_datum(type_oid, sort_op_oid, collation_oid, nulls_first, static_cast<int>(work_mem_kb),
                                 nullptr, // SortCoordinate (NULL = serial sort, not parallel)
                                 TUPLESORT_NONE // sortopt flags
    );
}

void PgSortRuntime::putDatum(void* sort_state, const uint64_t value, const bool is_null) {
    auto* tup_sort = static_cast<Tuplesortstate*>(sort_state);
    tuplesort_putdatum(tup_sort, value, is_null);
}

void PgSortRuntime::performSort(void* sort_state) {
    auto* tup_sort = static_cast<Tuplesortstate*>(sort_state);
    tuplesort_performsort(tup_sort);
}

bool PgSortRuntime::getDatum(void* sort_state, void* value_ptr, void* is_null_ptr) {
    auto* tup_sort = static_cast<Tuplesortstate*>(sort_state);
    auto* value = static_cast<uint64_t*>(value_ptr);
    auto* is_null = static_cast<bool*>(is_null_ptr);

    Datum datum;
    bool null_flag;

    // forward=true (ascending), copy=false (don't copy, we'll materialize into batch)
    const bool has_data = tuplesort_getdatum(tup_sort,
                                             true, // forward
                                             false, // copy
                                             &datum, &null_flag,
                                             nullptr // abbrev (abbreviated key, not needed)
    );

    if (has_data) {
        *value = datum;
        *is_null = null_flag;
    }

    return has_data;
}

void PgSortRuntime::end(void* sort_state) {
    auto* tup_sort = static_cast<Tuplesortstate*>(sort_state);
    tuplesort_end(tup_sort);
}

} // namespace runtime
