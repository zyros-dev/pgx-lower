#pragma once

#include <cstdint>

// PostgreSQL SPI stub function declarations
// These functions provide the interface to PostgreSQL's Server Programming Interface (SPI)
// for accessing table data within the MLIR JIT execution context

extern "C" {
    // Open a PostgreSQL table by name and return a handle
    void* pg_table_open(const char* table_name);
    
    // Get the next tuple from an open table handle
    // Returns a tuple handle or 0 if no more tuples
    int64_t pg_get_next_tuple(void* table_handle);
    
    // Extract a field value from a tuple
    // Returns the field value as an int32_t
    int32_t pg_extract_field(void* tuple_handle, int32_t field_index);
}