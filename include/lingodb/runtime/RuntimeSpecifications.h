#ifndef RUNTIME_RUNTIMESPECIFICATIONS_H
#define RUNTIME_RUNTIMESPECIFICATIONS_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

extern "C" {
#include "postgres.h"
}

namespace runtime {

enum class PhysicalType {
    BOOL,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    VARLEN32,
    NUMERIC_DATUM,
    INTERVAL,
};

struct ColumnLayout {
    size_t tuple_offset;
    size_t null_flag_offset;
    size_t value_offset;
    size_t value_size;
    PhysicalType phys_type;
    bool is_nullable;
    uint32_t pg_type_oid;
    int32_t pg_typmod;
    uint32_t pg_collation;
};

size_t get_physical_size(uint32_t type_oid);
PhysicalType get_physical_type(uint32_t type_oid);

using NumericDatumCarrier = Datum;

NumericDatumCarrier numeric_datum_to_carrier(Datum datum);
Datum numeric_datum_from_carrier(NumericDatumCarrier carrier);
void store_numeric_datum_carrier(uint8_t* dest, Datum datum);
NumericDatumCarrier load_numeric_datum_carrier(const uint8_t* src);

size_t extract_varlen32_string(const uint8_t* varlen32_data, char* dest, size_t max_len);

struct SortColumnInfo {
    const char* table_name;
    const char* column_name;
    uint32_t type_oid;
    int32_t typmod;
    uint32_t collation;
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
    uint32_t collation;
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
