#include "lingodb/runtime/LazyJoinHashtable.h"

extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
}

#include "pgx-lower/utility/logging.h"

runtime::LazyJoinHashtable* runtime::LazyJoinHashtable::create(size_t typeSize, uint64_t specPtr) {
    auto* ht = new LazyJoinHashtable(1024, typeSize);
    ht->spec = reinterpret_cast<const HashtableSpecification*>(specPtr);

    if (ht->spec) {
        ht->hashtable_context = AllocSetContextCreate(CurrentMemoryContext, "LazyJoinHashtableContext",
                                                      ALLOCSET_DEFAULT_SIZES);
        ht->compute_column_layouts();
    }

    return ht;
}

void runtime::LazyJoinHashtable::compute_column_layouts() {
    size_t offset = sizeof(Entry);
    for (int32_t i = 0; i < spec->num_key_columns; i++) {
        ColumnLayout layout;
        layout.pg_type_oid = spec->key_columns[i].type_oid;
        layout.phys_type = runtime::get_physical_type(layout.pg_type_oid);
        layout.is_nullable = true;

        layout.tuple_offset = offset;
        layout.null_flag_offset = offset;
        offset += 1; // null flag

        layout.value_offset = offset;
        layout.value_size = runtime::get_physical_size(layout.pg_type_oid);
        offset += layout.value_size;

        key_layouts_.push_back(layout);
    }

    for (int32_t i = 0; i < spec->num_value_columns; i++) {
        ColumnLayout layout;
        layout.pg_type_oid = spec->value_columns[i].type_oid;
        layout.phys_type = runtime::get_physical_type(layout.pg_type_oid);
        layout.is_nullable = true;

        layout.tuple_offset = offset;
        layout.null_flag_offset = offset;
        offset += 1; // null flag

        layout.value_offset = offset;
        layout.value_size = runtime::get_physical_size(layout.pg_type_oid);
        offset += layout.value_size;

        value_layouts_.push_back(layout);
    }
}

void runtime::LazyJoinHashtable::pack_entry_with_deep_copy(const uint8_t* keyData, const uint8_t* valueData,
                                                           uint8_t* entry) {
    PGX_IO(RUNTIME);

    const MemoryContext oldContext = MemoryContextSwitchTo(static_cast<MemoryContext>(hashtable_context));

    for (size_t i = 0; i < key_layouts_.size(); i++) {
        const auto& layout = key_layouts_[i];
        const uint8_t null_flag = keyData[layout.null_flag_offset];
        entry[layout.null_flag_offset] = null_flag;

        if (null_flag == 0) { // Not null
            if (layout.phys_type == PhysicalType::VARLEN32) {
                const uint32_t len_with_flag = *reinterpret_cast<const uint32_t*>(&keyData[layout.value_offset]);
                const size_t len = len_with_flag & ~0x80000000;
                const char* str_ptr = *reinterpret_cast<char* const*>(&keyData[layout.value_offset + 8]);

                char* new_str = static_cast<char*>(palloc(len + 1));
                memcpy(new_str, str_ptr, len);
                new_str[len] = '\0';

                *reinterpret_cast<uint32_t*>(&entry[layout.value_offset]) = len_with_flag;
                *reinterpret_cast<char**>(&entry[layout.value_offset + 8]) = new_str;

                PGX_LOG(RUNTIME, DEBUG, "  pack join key[%zu] string: len=%zu, ptr=%p", i, len, new_str);
            } else {
                memcpy(&entry[layout.value_offset], &keyData[layout.value_offset], layout.value_size);
            }
        }
    }

    for (size_t i = 0; i < value_layouts_.size(); i++) {
        const auto& layout = value_layouts_[i];
        const uint8_t null_flag = valueData[layout.null_flag_offset];
        entry[layout.null_flag_offset] = null_flag;

        if (null_flag == 0) { // Not null
            if (layout.phys_type == PhysicalType::VARLEN32) {
                const uint32_t len_with_flag = *reinterpret_cast<const uint32_t*>(&valueData[layout.value_offset]);
                const size_t len = len_with_flag & ~0x80000000;
                const char* str_ptr = *reinterpret_cast<char* const*>(&valueData[layout.value_offset + 8]);

                char* new_str = static_cast<char*>(palloc(len + 1));
                memcpy(new_str, str_ptr, len);
                new_str[len] = '\0';

                *reinterpret_cast<uint32_t*>(&entry[layout.value_offset]) = len_with_flag;
                *reinterpret_cast<char**>(&entry[layout.value_offset + 8]) = new_str;

                PGX_LOG(RUNTIME, DEBUG, "  pack join value[%zu] string: len=%zu, ptr=%p", i, len, new_str);
            } else {
                memcpy(&entry[layout.value_offset], &valueData[layout.value_offset], layout.value_size);
            }
        }
    }

    MemoryContextSwitchTo(oldContext);
}

void runtime::LazyJoinHashtable::appendEntryWithDeepCopy(size_t hashValue, size_t currentLen, const uint8_t* keyData,
                                                         const uint8_t* valueData) {
    PGX_IO(RUNTIME);

    // Note: LazyJoin stores {hash, entry} at MLIR level, not in C++ struct
    (void)hashValue;
    // Get pointer to next entry slot (don't increment len - MLIR handles that)
    uint8_t* entry = values.ptrAt<uint8_t>(currentLen);
    // Set next pointer (nullptr)
    *reinterpret_cast<Entry**>(entry) = nullptr;
    // Deep-copy key and value (handles all VarLen32 fields)
    if (spec && hashtable_context) {
        pack_entry_with_deep_copy(keyData, valueData, entry);
    } else {
        const size_t header_size = sizeof(Entry*);
        const size_t kv_size = entry_size - header_size;
        memcpy(entry + header_size, keyData, kv_size);
    }
}

void runtime::LazyJoinHashtable::resize() {
    values.resize();
}

void runtime::LazyJoinHashtable::finalize() {
    size_t htSize = std::max(nextPow2(values.getLen()), 1ul);
    htMask = htSize - 1;
    ht.setNewSize(htSize);
    for (size_t i = 0; i < values.getLen(); i++) {
        auto* entry = values.ptrAt<Entry>(i);
        size_t hash = (size_t)entry->next;
        auto pos = hash & htMask;
        auto* previousPtr = ht.at(pos);
        ht.at(pos) = runtime::tag(entry, previousPtr, hash);
        entry->next = previousPtr;
    }
}

void runtime::LazyJoinHashtable::destroy(LazyJoinHashtable* ht) {
    if (ht->hashtable_context) {
        MemoryContextDelete(static_cast<MemoryContext>(ht->hashtable_context));
    }
    delete ht;
}