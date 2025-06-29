#pragma once

#include <cstdint>

// Pure C interface that MLIR can call
// This interface must be implemented both in PostgreSQL extension and unit tests
extern "C" {

// MLIR Interface: Read next tuple for iteration control
// Returns: 1 = "we have a tuple", -2 = "end of table"
// Side effect: Preserves COMPLETE PostgreSQL tuple for later streaming
// Architecture: MLIR iterates, PostgreSQL handles all data types
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
}