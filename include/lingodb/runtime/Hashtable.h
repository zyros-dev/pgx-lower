#ifndef RUNTIME_HASHTABLE_H
#define RUNTIME_HASHTABLE_H
#include "lingodb/runtime/Vector.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/RuntimeSpecifications.h"
#include <vector>
#include <map>

namespace runtime {

struct HashtableMetadata {
    void* hashtable_context;
    const HashtableSpecification* spec;
    uint64_t entry_size;
};

extern std::map<void*, HashtableMetadata> g_hashtable_metadata;

class Hashtable {
    struct Entry {
        Entry* next;
        size_t hashValue;
    };
    runtime::FixedSizedBuffer<Entry*> ht;
    runtime::Vector values;

    Hashtable(size_t initialCapacity, size_t typeSize)
    : ht(initialCapacity * 2)
    , values(initialCapacity, typeSize) {
        g_hashtable_metadata[this] = HashtableMetadata{};
        g_hashtable_metadata[this].entry_size = typeSize;
    }

   public:
    void resize();
    void* appendEntryWithDeepCopy(size_t hashValue, size_t currentLen, void* keyPtr, void* valuePtr);
    static Hashtable* create(size_t typeSize, size_t initialCapacity, uint64_t specPtr);
    static void destroy(Hashtable*);
    void hex_dump_all() const;
};

} // end namespace runtime
#endif // RUNTIME_HASHTABLE_H
