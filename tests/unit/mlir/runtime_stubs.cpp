// Runtime stubs for unit testing
// These are minimal implementations to satisfy linker dependencies
// without pulling in the full PostgreSQL runtime

#include <string>
#include <memory>
#include <optional>
#include "runtime/metadata.h"

// Forward declarations needed
namespace mlir {
class TypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace runtime {

// Implement ColumnMetaData methods
const std::optional<size_t>& ColumnMetaData::getDistinctValues() const {
    return distinctValues;
}

void ColumnMetaData::setDistinctValues(const std::optional<size_t>& distinctValues) {
    this->distinctValues = distinctValues;
}

const ColumnType& ColumnMetaData::getColumnType() const {
    return columnType;
}

void ColumnMetaData::setColumnType(const ColumnType& columnType) {
    this->columnType = columnType;
}

// Implement the methods declared in metadata.h
std::shared_ptr<TableMetaData> TableMetaData::deserialize(std::string str) {
    // Minimal stub for unit testing
    auto result = std::make_shared<TableMetaData>();
    result->present = true;
    return result;
}

std::string TableMetaData::serialize(bool serializeSample) const {
    // Minimal stub for unit testing
    return "test_metadata";
}

bool TableMetaData::isPresent() const {
    // Minimal stub for unit testing
    return present;
}

} // namespace runtime

namespace mlir {
namespace util {

void populateUtilTypeConversionPatterns(TypeConverter& converter, RewritePatternSet& patterns) {
    // Minimal stub for unit testing
    // The real implementation would populate conversion patterns
}

} // namespace util
} // namespace mlir

//===----------------------------------------------------------------------===//
// Missing Runtime Function Stubs for JIT Engine
//===----------------------------------------------------------------------===//

extern "C" {

// DSA Runtime Functions
void* pgx_runtime_create_table_builder() { return nullptr; }
void pgx_runtime_append_i64(void*, int64_t) {}
void pgx_runtime_append_i64_direct(void*, int64_t) {}
void pgx_runtime_append_nullable_i64(void*, int64_t, bool) {}
void pgx_runtime_append_null(void*) {}
bool pgx_runtime_table_next_row(void*) { return false; }

// PostgreSQL SPI Functions
void* pg_table_open(const char*) { return nullptr; }
int64_t pg_get_next_tuple(void*) { return 0; }
int32_t pg_extract_field(void*, int32_t) { return 0; }
void pg_store_result(void*) {}
void pg_store_result_i32(int32_t) {}
void pg_store_result_i64(int64_t) {}
void pg_store_result_f64(double) {}
void pg_store_result_text(const char*) {}

// Memory Management Functions
void* pgx_exec_alloc_state_raw(size_t) { return nullptr; }
void pgx_exec_free_state(void*) {}
void pgx_exec_set_tuple_count(void*, int64_t) {}
int64_t pgx_exec_get_tuple_count(void*) { return 0; }
void* pgx_threadlocal_create() { return nullptr; }
void* pgx_threadlocal_get(void*) { return nullptr; }
void pgx_threadlocal_merge(void*, void*) {}

// Data Source Functions
void* pgx_datasource_get(void*) { return nullptr; }
void* pgx_datasource_iteration_init(void*) { return nullptr; }
void* pgx_datasource_iteration_iterate(void*) { return nullptr; }
void* pgx_buffer_create_zeroed(size_t) { return nullptr; }
void* pgx_buffer_iterate(void*) { return nullptr; }
void* pgx_growing_buffer_create() { return nullptr; }
void pgx_growing_buffer_insert(void*, void*) {}

// Table Access Functions
void* open_postgres_table(const char*) { return nullptr; }
int64_t read_next_tuple_from_table(void*) { return 0; }
void close_postgres_table(void*) {}

// Field Access Functions
int32_t get_int_field(void*, int32_t, bool* is_null) { 
    if (is_null) *is_null = true; 
    return 0; 
}
int64_t get_text_field(void*, int32_t, bool* is_null) { 
    if (is_null) *is_null = true; 
    return 0; 
}
double get_numeric_field(void*, int32_t, bool* is_null) { 
    if (is_null) *is_null = true; 
    return 0.0; 
}

// Result Storage Functions
bool add_tuple_to_result(int64_t) { return false; }
void store_int_result(int32_t, int32_t, bool) {}
void store_bigint_result(int32_t, int64_t, bool) {}
void store_text_result(int32_t, const char*, bool) {}
void store_field_as_datum(int32_t, int64_t, int32_t) {}
void mark_results_ready_for_streaming() {}
void prepare_computed_results(int32_t) {}

// MLIR Field Access Functions
int32_t get_int_field_mlir(int64_t, int32_t) { return 0; }

} // extern "C"