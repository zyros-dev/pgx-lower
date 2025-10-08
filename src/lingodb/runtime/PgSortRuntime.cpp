#include "lingodb/runtime/PgSortRuntime.h"

extern "C" {
#include "postgres.h"
#include "utils/tuplesort.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_collation.h"
#include "utils/sortsupport.h"
#include "miscadmin.h"
}

namespace runtime {

// When we have a single column we don't need to know the sort state, but we still need some wrapper object
struct SortStateWrapper {
    Tuplesortstate* tuplesort;
    TupleDesc tupledesc;
};

void* PgSortRuntime::beginDatum(const uint32_t type_oid, const uint32_t sort_op_oid, const uint32_t collation_oid,
                                const bool nulls_first, const uint32_t work_mem_kb) {
    return tuplesort_begin_datum(type_oid, sort_op_oid, collation_oid, nulls_first, static_cast<int>(work_mem_kb),
                                 nullptr, // SortCoordinate (NULL = serial sort, not parallel)
                                 TUPLESORT_NONE);
}

void PgSortRuntime::putDatum(void* sort_state, const uint64_t value, const bool is_null) {
    auto* tup_sort = static_cast<Tuplesortstate*>(sort_state);
    tuplesort_putdatum(tup_sort, value, is_null);
}

void PgSortRuntime::performSort(void* sort_state) {
    auto* wrapper = static_cast<SortStateWrapper*>(sort_state);
    if (wrapper->tuplesort) {
        tuplesort_performsort(wrapper->tuplesort);
    } else {
        tuplesort_performsort(static_cast<Tuplesortstate*>(sort_state));
    }
}

bool PgSortRuntime::getDatum(void* sort_state, void* value_ptr, void* is_null_ptr) {
    auto* tup_sort = static_cast<Tuplesortstate*>(sort_state);
    auto* value = static_cast<uint64_t*>(value_ptr);
    auto* is_null = static_cast<bool*>(is_null_ptr);

    Datum datum;
    bool null_flag;

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

void* PgSortRuntime::beginHeapSort(const uint32_t* type_oids, const int32_t* typmods, const uint32_t* sort_op_oids,
                                   const int32_t* directions, const int32_t num_cols) {
    const TupleDesc tup_desc = CreateTemplateTupleDesc(num_cols);
    for (int i = 0; i < num_cols; i++) {
        TupleDescInitEntry(tup_desc,
                           i + 1, // attribute number (1-based)
                           nullptr, // attribute name (not needed for sorting)
                           type_oids[i], typmods[i],
                           0 // dimension (not an array)
        );
    }

    // Build arrays for tuplesort_begin_heap
    const auto att_nums = static_cast<AttrNumber*>(palloc(num_cols * sizeof(AttrNumber)));
    const auto sort_operators = static_cast<Oid*>(palloc(num_cols * sizeof(Oid)));
    const auto sort_collations = static_cast<Oid*>(palloc(num_cols * sizeof(Oid)));
    const auto nulls_first = static_cast<bool*>(palloc(num_cols * sizeof(bool)));

    for (int i = 0; i < num_cols; i++) {
        att_nums[i] = i + 1;
        sort_operators[i] = sort_op_oids[i];
        sort_collations[i] = DEFAULT_COLLATION_OID;
        nulls_first[i] = (directions[i] == 0); // DESC = nulls first
    }

    Tuplesortstate* state = tuplesort_begin_heap(tup_desc, num_cols, att_nums, sort_operators, sort_collations,
                                                 nulls_first,
                                                 work_mem, // work_mem setting (global PostgreSQL variable)
                                                 nullptr, TUPLESORT_NONE);

    const auto wrapper = static_cast<SortStateWrapper*>(palloc(sizeof(SortStateWrapper)));
    wrapper->tuplesort = state;
    wrapper->tupledesc = tup_desc;
    return wrapper;
}

void PgSortRuntime::putHeapTuple(void* sort_state, const uintptr_t* datums, const bool* nulls, int32_t num_cols) {
    const auto* wrapper = static_cast<SortStateWrapper*>(sort_state);
    const HeapTuple tuple = heap_form_tuple(wrapper->tupledesc, datums, nulls);
    tuplesort_putheaptuple(wrapper->tuplesort, tuple);
    heap_freetuple(tuple);
}

bool PgSortRuntime::getHeapTuple(void* sort_state, uintptr_t* datums_out, bool* nulls_out, int32_t num_cols) {
    const auto* wrapper = static_cast<SortStateWrapper*>(sort_state);
    const HeapTuple tuple = tuplesort_getheaptuple(wrapper->tuplesort, true);
    if (!tuple) {
        return false;
    }
    heap_deform_tuple(tuple, wrapper->tupledesc, reinterpret_cast<Datum*>(datums_out), nulls_out);
    return true;
}

} // namespace runtime
