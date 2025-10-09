#include "lingodb/runtime/PgSortRuntime.h"

#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "utils/tuplesort.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_collation.h"
#include "utils/sortsupport.h"
#include "utils/datum.h"
#include "miscadmin.h"
}

namespace runtime {

// When we have a single column we don't need to know the sort state, but we still need some wrapper object
struct SortStateWrapper {
    Tuplesortstate* tuplesort;
    TupleDesc tupledesc;
    Datum* prev_datums; // Track previous datums for memory management
    int32_t num_cols;
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
    PGX_LOG(RUNTIME, DEBUG, "performSort: starting sort");
    auto* wrapper = static_cast<SortStateWrapper*>(sort_state);
    if (wrapper->tuplesort) {
        PGX_LOG(RUNTIME, DEBUG, "performSort: calling tuplesort_performsort on wrapper");
        tuplesort_performsort(wrapper->tuplesort);
        PGX_LOG(RUNTIME, DEBUG, "performSort: sort completed");
    } else {
        PGX_LOG(RUNTIME, DEBUG, "performSort: calling tuplesort_performsort directly");
        tuplesort_performsort(static_cast<Tuplesortstate*>(sort_state));
        PGX_LOG(RUNTIME, DEBUG, "performSort: sort completed");
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

void* PgSortRuntime::beginHeapSort(const uint32_t* all_type_oids, const int32_t* all_typmods, int32_t num_total_cols,
                                   const int32_t* sort_key_indices, const uint32_t* sort_op_oids,
                                   const int32_t* sort_directions, int32_t num_sort_keys) {
    PGX_LOG(RUNTIME, DEBUG, "beginHeapSort: total_cols=%d, sort_keys=%d", num_total_cols, num_sort_keys);
    PGX_LOG(RUNTIME, DEBUG, "beginHeapSort: first type_oid=%u, first sort_op_oid=%u", all_type_oids[0], sort_op_oids[0]);

    // Build TupleDesc for ALL columns
    const TupleDesc tup_desc = CreateTemplateTupleDesc(num_total_cols);
    for (int i = 0; i < num_total_cols; i++) {
        TupleDescInitEntry(tup_desc,
                           i + 1, // attribute number (1-based)
                           nullptr, // attribute name (not needed for sorting)
                           all_type_oids[i], all_typmods[i],
                           0 // dimension (not an array)
        );
    }
    PGX_LOG(RUNTIME, DEBUG, "beginHeapSort: TupleDesc created with %d columns", num_total_cols);

    // Build arrays for sort keys only
    const auto att_nums = static_cast<AttrNumber*>(palloc(num_sort_keys * sizeof(AttrNumber)));
    const auto sort_operators = static_cast<Oid*>(palloc(num_sort_keys * sizeof(Oid)));
    const auto sort_collations = static_cast<Oid*>(palloc(num_sort_keys * sizeof(Oid)));
    const auto nulls_first = static_cast<bool*>(palloc(num_sort_keys * sizeof(bool)));

    for (int i = 0; i < num_sort_keys; i++) {
        att_nums[i] = sort_key_indices[i] + 1; // Convert to 1-based
        sort_operators[i] = sort_op_oids[i];
        sort_collations[i] = DEFAULT_COLLATION_OID;
        nulls_first[i] = (sort_directions[i] == 0); // DESC = nulls first
    }

    PGX_LOG(RUNTIME, DEBUG, "beginHeapSort: calling tuplesort_begin_heap with nkeys=%d", num_sort_keys);
    Tuplesortstate* state = tuplesort_begin_heap(tup_desc, num_sort_keys, att_nums, sort_operators, sort_collations,
                                                 nulls_first,
                                                 work_mem, // work_mem setting (global PostgreSQL variable)
                                                 nullptr, TUPLESORT_NONE);

    const auto wrapper = static_cast<SortStateWrapper*>(palloc(sizeof(SortStateWrapper)));
    wrapper->tuplesort = state;
    wrapper->tupledesc = tup_desc;
    wrapper->prev_datums = nullptr;
    wrapper->num_cols = num_total_cols;
    PGX_LOG(RUNTIME, DEBUG, "beginHeapSort: completed successfully");
    return wrapper;
}

void PgSortRuntime::putHeapTuple(void* sort_state, const uintptr_t* datums, const bool* nulls, int32_t num_cols) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: num_cols=%d", num_cols);
    const auto* wrapper = static_cast<SortStateWrapper*>(sort_state);
    PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: TupleDesc has %d columns", wrapper->tupledesc->natts);

    for (int i = 0; i < num_cols; i++) {
        PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: col[%d] datum=%lu null=%d typeOid=%u", i, datums[i], nulls[i],
                TupleDescAttr(wrapper->tupledesc, i)->atttypid);
    }

    PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: About to call heap_form_tuple");
    const HeapTuple tuple = heap_form_tuple(wrapper->tupledesc, datums, nulls);
    PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: heap_form_tuple succeeded, about to call tuplesort_putheaptuple");
    tuplesort_putheaptuple(wrapper->tuplesort, tuple);
    PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: tuplesort_putheaptuple succeeded, about to free tuple");
    heap_freetuple(tuple);
    PGX_LOG(RUNTIME, DEBUG, "putHeapTuple: completed successfully");
}

bool PgSortRuntime::getHeapTuple(void* sort_state, uintptr_t* datums_out, bool* nulls_out, int32_t num_cols) {
    PGX_LOG(RUNTIME, TRACE, "getHeapTuple: num_cols=%d", num_cols);
    auto* wrapper = static_cast<SortStateWrapper*>(sort_state);

    // Free previous datums if they exist
    if (wrapper->prev_datums) {
        for (int i = 0; i < wrapper->num_cols; i++) {
            if (wrapper->prev_datums[i] != 0) {
                Form_pg_attribute attr = TupleDescAttr(wrapper->tupledesc, i);
                if (!attr->attbyval) {
                    pfree(DatumGetPointer(wrapper->prev_datums[i]));
                }
            }
        }
        pfree(wrapper->prev_datums);
        wrapper->prev_datums = nullptr;
    }

    const HeapTuple tuple = tuplesort_getheaptuple(wrapper->tuplesort, true);
    if (!tuple) {
        return false;
    }

    // Deform tuple into temporary arrays
    Datum temp_datums[num_cols];
    bool temp_nulls[num_cols];
    heap_deform_tuple(tuple, wrapper->tupledesc, temp_datums, temp_nulls);

    // Allocate storage for tracking datums
    wrapper->prev_datums = static_cast<Datum*>(palloc0(num_cols * sizeof(Datum)));

    // Copy datums to output, handling pass-by-reference types
    for (int i = 0; i < num_cols; i++) {
        nulls_out[i] = temp_nulls[i];
        if (!temp_nulls[i]) {
            // Get type info to determine if pass-by-value or pass-by-reference
            Form_pg_attribute attr = TupleDescAttr(wrapper->tupledesc, i);
            if (attr->attbyval) {
                // Pass-by-value: just copy the datum
                datums_out[i] = temp_datums[i];
                wrapper->prev_datums[i] = 0;
            } else {
                // Pass-by-reference: need to copy the data
                datums_out[i] = datumCopy(temp_datums[i], false, attr->attlen);
                wrapper->prev_datums[i] = datums_out[i];
            }
        } else {
            datums_out[i] = 0;
            wrapper->prev_datums[i] = 0;
        }
    }

    return true;
}

} // namespace runtime
