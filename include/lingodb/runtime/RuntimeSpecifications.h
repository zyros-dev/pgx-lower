#ifndef RUNTIME_RUNTIMESPECIFICATIONS_H
#define RUNTIME_RUNTIMESPECIFICATIONS_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace runtime {

enum class PhysicalType {
    BOOL,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    VARLEN32,
    DECIMAL128,
};

struct ColumnLayout {
    size_t tuple_offset;
    size_t null_flag_offset;
    size_t value_offset;
    size_t value_size;
    PhysicalType phys_type;
    bool is_nullable;
    uint32_t pg_type_oid;
};

size_t get_physical_size(uint32_t type_oid);
PhysicalType get_physical_type(uint32_t type_oid);

size_t extract_varlen32_string(const uint8_t* i128_data, char* dest, size_t max_len);

struct SortColumnInfo {
    const char* table_name;
    const char* column_name;
    uint32_t type_oid;
    int32_t typmod;
    bool is_nullable;
};

struct SortSpecification {
    SortColumnInfo* columns;
    int32_t num_columns;
    int32_t* sort_key_indices;
    uint32_t* sort_operators;
    uint32_t* collations;
    bool* nulls_first;
    int32_t num_sort_keys;
};

struct HashtableColumnInfo {
    const char* table_name;
    const char* column_name;
    uint32_t type_oid;
    int32_t typmod;
    bool is_nullable;
};

struct HashtableSpecification {
    HashtableColumnInfo* key_columns;
    int32_t num_key_columns;
    HashtableColumnInfo* value_columns;
    int32_t num_value_columns;
};

} // namespace runtime

#endif // RUNTIME_RUNTIMESPECIFICATIONS_H
