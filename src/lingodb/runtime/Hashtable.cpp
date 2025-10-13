#include "lingodb/runtime/Hashtable.h"

extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
}

#include "pgx-lower/utility/logging.h"

std::map<void*, runtime::HashtableMetadata> runtime::g_hashtable_metadata;

static void hex_dump(const char* label, const void* ptr, size_t length) {
    const auto bytes = static_cast<const uint8_t*>(ptr);
    std::string output = std::string(label) + " hex dump (" + std::to_string(length) + " bytes):\n";

    for (size_t i = 0; i < length; i += 16) {
        char line[128];
        const size_t remaining = length - i;
        const size_t chunk_size = remaining < 16 ? remaining : 16;

        char* p = line;
        p += sprintf(p, "  [%02zu-%02zu]: ", i, i + chunk_size - 1);

        for (size_t j = 0; j < chunk_size; j++) {
            p += sprintf(p, "%02x ", bytes[i + j]);
            if (j == 7 && chunk_size > 8) {
                p += sprintf(p, " ");
            }
        }

        output += line;
        output += "\n";
    }

    PGX_LOG(RUNTIME, DEBUG, "%s", output.c_str());
}

void runtime::Hashtable::hex_dump_all() const {
    const auto& meta = g_hashtable_metadata[const_cast<Hashtable*>(this)];

    PGX_LOG(RUNTIME, DEBUG, "=== Hashtable full hex dump at %p ===", static_cast<const void*>(this));

    PGX_LOG(RUNTIME, DEBUG, "Metadata: entry_size=%zu, spec=%p, context=%p", meta.entry_size,
            static_cast<const void*>(meta.spec), meta.hashtable_context);

    const size_t bucket_count = values.getCap() * 2;
    PGX_LOG(RUNTIME, DEBUG, "Bucket array: %zu buckets", bucket_count);
    hex_dump("Bucket array", ht.ptr, bucket_count * sizeof(Entry*));

    const size_t entry_count = values.getLen();
    const size_t entry_size = meta.entry_size;
    PGX_LOG(RUNTIME, DEBUG, "Entry storage: %zu entries, %zu bytes each", entry_count, entry_size);

    for (size_t i = 0; i < entry_count; i++) {
        const uint8_t* entry = values.getPtr() + (i * meta.entry_size);
        char label[64];
        sprintf(label, "Entry[%zu]", i);
        hex_dump(label, entry, entry_size);
    }

    PGX_LOG(RUNTIME, DEBUG, "=== End hashtable hex dump ===");
}

runtime::Hashtable* runtime::Hashtable::create(size_t typeSize, size_t initialCapacity, uint64_t specPtr) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "Hashtable::create - typeSize=%zu, initialCapacity=%zu, specPtr=0x%lx", typeSize,
            initialCapacity, specPtr);

    auto* ht = new (malloc(sizeof(Hashtable) + typeSize)) Hashtable(initialCapacity, typeSize);

    auto& meta = g_hashtable_metadata[ht];
    meta.spec = reinterpret_cast<const HashtableSpecification*>(specPtr);
    meta.entry_size = typeSize;

    PGX_LOG(RUNTIME, DEBUG, "Hashtable::create - ht=%p, spec=%p", static_cast<void*>(ht),
            static_cast<const void*>(meta.spec));

    if (meta.spec) {
        PGX_LOG(RUNTIME, DEBUG,
                "Hashtable::create - spec is present, num_key_columns=%d, num_value_columns=%d, entry_size=%zu",
                meta.spec->num_key_columns, meta.spec->num_value_columns, meta.entry_size);
        // ReSharper disable once CppStaticAssertFailure
        meta.hashtable_context = AllocSetContextCreate(CurrentMemoryContext, "HashtableContext", ALLOCSET_DEFAULT_SIZES);
    }

    PGX_LOG(RUNTIME, DEBUG, "Hashtable::create - RETURNING ht=%p with spec=%p", static_cast<void*>(ht),
            static_cast<const void*>(meta.spec));

    ht->hex_dump_all();
    return ht;
}

void* runtime::Hashtable::appendEntryWithDeepCopy(size_t hashValue, size_t currentLen, void* keyPtr, void* valuePtr, size_t key_size, size_t value_size) {
    PGX_IO(RUNTIME);

    const auto& meta = g_hashtable_metadata[this];

    uint8_t* entry = values.ptrAt<uint8_t>(currentLen);

    PGX_LOG(RUNTIME, DEBUG, "appendEntryWithDeepCopy: entry=%p, hashValue=%zu, currentLen=%zu, entry_size=%zu, key_size=%zu, value_size=%zu",
            static_cast<void*>(entry), hashValue, currentLen, meta.entry_size, key_size, value_size);

    // Write Entry header: [0-7]=next, [8-15]=hash
    *reinterpret_cast<Entry**>(entry) = nullptr;
    *reinterpret_cast<size_t*>(entry + sizeof(Entry*)) = hashValue;

    const size_t header_size = sizeof(Entry);  // 16 bytes
    uint8_t* kv_region = entry + header_size;

    memcpy(kv_region, keyPtr, key_size);
    memcpy(kv_region + key_size, valuePtr, value_size);

    PGX_LOG(RUNTIME, DEBUG, "Copied key+value: key_size=%zu, value_size=%zu, total=%zu", key_size, value_size, key_size + value_size);

    // Deep copy strings if present
    if (meta.spec && meta.hashtable_context) {
        PGX_LOG(RUNTIME, DEBUG, "RUNTIME: Using HashtableSpec at %p: num_keys=%d, num_vals=%d",
                    meta.spec, meta.spec->num_key_columns, meta.spec->num_value_columns);
        MemoryContext oldContext = MemoryContextSwitchTo(static_cast<MemoryContext>(meta.hashtable_context));

        size_t offset = 0;
        for (int32_t i = 0; i < meta.spec->num_key_columns; i++) {
            const auto& col = meta.spec->key_columns[i];
            const uint32_t type_oid = col.type_oid;
            PGX_LOG(RUNTIME, DEBUG, "RUNTIME: Reading hashtable key_column[%d]: type_oid=%u, nullable=%d",
                        i, type_oid, col.is_nullable);
            const size_t col_size = get_physical_size(type_oid);

            if (type_oid == VARCHAROID || type_oid == TEXTOID) {
                // VarLen32 i128 layout - TWO cases:
                // Case 1 (lazy flag SET): Runtime pointer-based string from table scan
                //   bytes[0-3]:   len | 0x80000000
                //   bytes[8-15]:  valid pointer to string data
                // Case 2 (lazy flag CLEAR): MLIR inlined constant from CASE/literal
                //   bytes[0-3]:   len (no flag)
                //   bytes[4-15]:  inlined string data
                uint8_t* col_data = kv_region + offset;
                uint8_t* i128_data = col.is_nullable ? (col_data + 1) : col_data;

                const uint32_t len_with_flag = *reinterpret_cast<uint32_t*>(i128_data);
                const bool is_lazy = (len_with_flag & 0x80000000u) != 0;
                const uint32_t len = len_with_flag & ~0x80000000u;

                if (is_lazy) {
                    // Case 1: Runtime pointer-based string - must deep-copy
                    const uint64_t ptr_val = *reinterpret_cast<uint64_t*>(i128_data + 8);
                    char* src_str = reinterpret_cast<char*>(ptr_val);

                    PGX_LOG(RUNTIME, DEBUG, "Deep copying key string[%d] (lazy): len=%u, src=%p", i, len, static_cast<void*>(src_str));

                    char* new_str = static_cast<char*>(palloc(len + 1));
                    memcpy(new_str, src_str, len);
                    new_str[len] = '\0';

                    // Update pointer in i128
                    *reinterpret_cast<uint64_t*>(i128_data + 8) = reinterpret_cast<uint64_t>(new_str);

                    PGX_LOG(RUNTIME, DEBUG, "  Copied to %p: '%s'", static_cast<void*>(new_str), new_str);
                } else {
                    PGX_LOG(RUNTIME, DEBUG, "Key string[%d] (inlined): len=%u, no deep copy needed", i, len);
                }
            }

            offset += col.is_nullable ? (1 + col_size) : col_size;
        }

        // Process value columns
        offset = key_size;  // Start after keys
        for (int32_t i = 0; i < meta.spec->num_value_columns; i++) {
            const auto& col = meta.spec->value_columns[i];
            const uint32_t type_oid = col.type_oid;
            PGX_LOG(RUNTIME, DEBUG, "RUNTIME: Reading hashtable value_column[%d]: type_oid=%u, nullable=%d",
                        i, type_oid, col.is_nullable);
            const size_t col_size = get_physical_size(type_oid);

            if (type_oid == VARCHAROID || type_oid == TEXTOID) {
                uint8_t* col_data = kv_region + offset;
                uint8_t* i128_data = col.is_nullable ? (col_data + 1) : col_data;  // Skip nullable byte if present

                const uint32_t len_with_flag = *reinterpret_cast<uint32_t*>(i128_data);
                const bool is_lazy = (len_with_flag & 0x80000000u) != 0;
                const uint32_t len = len_with_flag & ~0x80000000u;

                if (is_lazy) {
                    // Runtime pointer-based string - must deep-copy
                    const uint64_t ptr_val = *reinterpret_cast<uint64_t*>(i128_data + 8);
                    char* src_str = reinterpret_cast<char*>(ptr_val);

                    PGX_LOG(RUNTIME, DEBUG, "Deep copying value string[%d] (lazy): len=%u, src=%p", i, len, static_cast<void*>(src_str));

                    char* new_str = static_cast<char*>(palloc(len + 1));
                    memcpy(new_str, src_str, len);
                    new_str[len] = '\0';

                    *reinterpret_cast<uint64_t*>(i128_data + 8) = reinterpret_cast<uint64_t>(new_str);

                    PGX_LOG(RUNTIME, DEBUG, "  Copied to %p: '%s'", static_cast<void*>(new_str), new_str);
                } else {
                    PGX_LOG(RUNTIME, DEBUG, "Value string[%d] (inlined): len=%u, no deep copy needed", i, len);
                }
            }

            offset += col.is_nullable ? (1 + col_size) : col_size;
        }

        MemoryContextSwitchTo(oldContext);
    }

    hex_dump("Final entry", entry, meta.entry_size);

    return entry;
}

void runtime::Hashtable::resize() {
    const auto& meta = g_hashtable_metadata[this];
    PGX_LOG(RUNTIME, DEBUG, "Hashtable::resize - BEFORE resize: this=%p, spec=%p", static_cast<void*>(this),
            static_cast<const void*>(meta.spec));
    PGX_IO(RUNTIME);
    hex_dump_all();

    const size_t oldHtSize = values.getCap() * 2;
    const size_t newHtSize = oldHtSize * 2;
    values.resize();
    ht.setNewSize(newHtSize);

    PGX_LOG(RUNTIME, DEBUG, "Hashtable::resize - AFTER resize: this=%p, spec=%p", static_cast<void*>(this),
            static_cast<const void*>(meta.spec));
    const size_t hashMask = newHtSize - 1;
    for (size_t i = 0; i < values.getLen(); i++) {
        auto* entry = values.ptrAt<Entry>(i);
        const auto pos = entry->hashValue & hashMask;
        auto* previousPtr = ht.at(pos);
        ht.at(pos) = entry;
        entry->next = previousPtr;
    }
}

void runtime::Hashtable::destroy(Hashtable* ht) {
    PGX_IO(RUNTIME);
    ht->hex_dump_all();
    const auto& meta = g_hashtable_metadata[ht];
    if (meta.hashtable_context) {
        MemoryContextDelete(static_cast<MemoryContext>(meta.hashtable_context));
    }
    g_hashtable_metadata.erase(ht);
    ht->~Hashtable();
    free(ht);
}