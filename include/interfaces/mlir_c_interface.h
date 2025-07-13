#pragma once

#include <cstdint>

// Pure C interface that MLIR can call
// This interface must be implemented both in PostgreSQL extension and unit tests
extern "C" {

// MLIR Interface: Read next tuple for iteration control
// Returns: tuple pointer as int64_t if valid tuple, 0 if end of table
auto read_next_tuple_from_table(void* tableHandle) -> int64_t;

// MLIR Interface: Stream complete PostgreSQL tuple to output
// The 'value' parameter is ignored - it's just MLIR's iteration signal
auto add_tuple_to_result(int64_t value) -> bool;

auto open_postgres_table(const char* tableName) -> void*;

void close_postgres_table(void* tableHandle);

// Legacy interface for simple tuple access (kept for compatibility)
auto get_next_tuple() -> int64_t;

// Typed field access functions for PostgreSQL dialect
auto get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int32_t;
auto get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) -> int64_t;
auto get_numeric_field(void* tuple_handle, int32_t field_index, bool* is_null) -> double;

// Result storage functions for expressions
void store_int_result(int32_t columnIndex, int32_t value, bool isNull);
void store_bool_result(int32_t columnIndex, bool value, bool isNull);
void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
void store_text_result(int32_t columnIndex, const char* value, bool isNull);
void prepare_computed_results(int32_t numColumns);

// Aggregate functions
int64_t sum_aggregate(void* table_handle);
}