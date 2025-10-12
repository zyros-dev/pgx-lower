#ifndef RUNTIME_LAZYJOINHASHTABLE_H
#define RUNTIME_LAZYJOINHASHTABLE_H
#include "lingodb/runtime/Vector.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/RuntimeSpecifications.h"
#include <vector>

namespace runtime {
class LazyJoinHashtable {
    struct Entry {
        Entry* next;
    };
    runtime::FixedSizedBuffer<Entry*> ht;
    size_t htMask;
    runtime::Vector values;

    const HashtableSpecification* spec;
    std::vector<ColumnLayout> key_layouts_;
    std::vector<ColumnLayout> value_layouts_;
    void* hashtable_context;
    size_t entry_size;

    LazyJoinHashtable(size_t initial, size_t typeSize)
    : ht(0)
    , htMask(0)
    , values(initial, typeSize)
    , spec(nullptr)
    , hashtable_context(nullptr)
    , entry_size(typeSize) {}
    static uint64_t nextPow2(uint64_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
        return v;
    }

    void compute_column_layouts();
    void pack_entry_with_deep_copy(const uint8_t* keyData, const uint8_t* valueData, uint8_t* entry);

   public:
    static LazyJoinHashtable* create(size_t typeSize, uint64_t specPtr);
    void appendEntryWithDeepCopy(size_t hashValue, size_t currentLen, const uint8_t* keyData, const uint8_t* valueData);
    void finalize();
    void resize();
    static void destroy(LazyJoinHashtable*);
};
} // end namespace runtime
#endif // RUNTIME_LAZYJOINHASHTABLE_H
