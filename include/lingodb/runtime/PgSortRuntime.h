#ifndef RUNTIME_PGSORTRUNTIME_H
#define RUNTIME_PGSORTRUNTIME_H

#include "RuntimeSpecifications.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace runtime {

class PgSortState {
    size_t tupleSize;
    size_t tupleCount;
    size_t capacity;
    const SortSpecification* spec; // (not owned - controlled by postgres transaction memory context)
    std::vector<ColumnLayout> column_layouts_;

    void* sortcontext;
    void* tupdesc;
    void* sortstate;
    void* input_slot;
    void* output_slot;
    bool sorted;
    size_t fetch_index;

    void build_tuple_desc();
    void init_tuplesort();
    void compute_column_layouts();
    void unpack_mlir_to_datums(const uint8_t* mlir_tuple, void* values, bool* isnull) const;
    void pack_datums_to_mlir(void* values, const bool* isnull, uint8_t* mlir_tuple) const;
    void log_specification(const char* context) const;

   public:
    PgSortState(const SortSpecification* spec, size_t tupleSize, size_t initialCapacity);
    void appendTuple(const uint8_t* tupleData);
    void flushTuples() const;
    void performSort();
    void* getNextTuple();
    static PgSortState* create(size_t tupleSize, uint64_t specPtr);
    static void destroy(PgSortState* state);

    ~PgSortState();
};

} // namespace runtime

#endif // RUNTIME_PGSORTRUNTIME_H
