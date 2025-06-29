#pragma once

#include <cstdint>

// Pure C interface that MLIR can call
// This interface must be implemented both in PostgreSQL extension and unit tests
extern "C" {

// MLIR Interface: Read next tuple for iteration control
// Returns: 1 = "we have a tuple", -2 = "end of table"
// Side effect: Preserves COMPLETE PostgreSQL tuple for later streaming
// Architecture: MLIR iterates, PostgreSQL handles all data types
int64_t read_next_tuple_from_table(void* tableHandle);

// MLIR Interface: Stream complete PostgreSQL tuple to output
// The 'value' parameter is ignored - it's just MLIR's iteration signal
bool add_tuple_to_result(int64_t value);

void* open_postgres_table(const char* tableName);

void close_postgres_table(void* tableHandle);

// Legacy interface for simple tuple access (kept for compatibility)
int64_t get_next_tuple();

// Typed field access functions for PostgreSQL dialect
int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null);
int64_t get_text_field(void* tuple_handle, int32_t field_index, bool* is_null);
}