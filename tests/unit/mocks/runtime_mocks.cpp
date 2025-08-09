// Mock implementations of runtime functions for unit tests
// These allow the ExecutionEngine to register symbols without requiring full PostgreSQL runtime

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C" {

// DSA Runtime Functions (mocks)
void* pgx_runtime_create_table_builder(const char* schema) {
    return reinterpret_cast<void*>(0x1000);
}

void pgx_runtime_append_i64(void* builder, size_t col_idx, int64_t value) {
    // Mock implementation
}

void pgx_runtime_append_i64_direct(void* builder, int64_t value) {
    // Mock implementation
}

void pgx_runtime_append_nullable_i64(void* builder, bool is_null, int64_t value) {
    // Mock implementation
}

void pgx_runtime_append_null(void* builder, size_t col_idx) {
    // Mock implementation
}

void pgx_runtime_table_next_row(void* builder) {
    // Mock implementation
}

// Note: Many of these functions are already defined in test_helpers.cpp
// Only define the ones that are missing

// PostgreSQL runtime support functions (mocks)
void* pgx_exec_alloc_state_raw(int64_t size) {
    // Use dynamic allocation for test isolation
    void* buffer = std::malloc(size > 0 ? size : 1);
    if (buffer) {
        std::memset(buffer, 0, size > 0 ? size : 1);
    }
    return buffer;
}

void pgx_exec_free_state(void* state) {
    // Actually free the memory for test isolation
    std::free(state);
}

void pgx_exec_set_tuple_count(void* exec_context, int64_t count) {
    // Mock implementation
}

int64_t pgx_exec_get_tuple_count(void* exec_context) {
    return 0;
}

void* pgx_threadlocal_create(int64_t size) {
    // Use dynamic allocation for test isolation
    void* buffer = std::malloc(size > 0 ? size : 1);
    if (buffer) {
        std::memset(buffer, 0, size > 0 ? size : 1);
    }
    return buffer;
}

void* pgx_threadlocal_get(void* tls) {
    return tls;
}

void pgx_threadlocal_merge(void* dest, void* src) {
    // Mock implementation
}

// Data source and buffer operations (mocks)
void* pgx_datasource_get(void* table_ref) {
    return reinterpret_cast<void*>(0x3000);
}

void* pgx_datasource_iteration_init(void* datasource, int64_t start, int64_t end) {
    return reinterpret_cast<void*>(0x3100);
}

int8_t pgx_datasource_iteration_iterate(void* iteration, void** row_out) {
    return 0; // No more rows
}

void* pgx_buffer_create_zeroed(int64_t size) {
    // Use dynamic allocation for test isolation
    void* buffer = std::calloc(1, size > 0 ? size : 1);
    return buffer;
}

void* pgx_buffer_iterate(void* buffer, int64_t index) {
    return reinterpret_cast<char*>(buffer) + index;
}

void* pgx_growing_buffer_create(int64_t initial_capacity) {
    // Use dynamic allocation for test isolation
    void* buffer = std::malloc(initial_capacity > 0 ? initial_capacity : 1024);
    if (buffer) {
        std::memset(buffer, 0, initial_capacity > 0 ? initial_capacity : 1024);
    }
    return buffer;
}

void pgx_growing_buffer_insert(void* buffer, void* value, int64_t value_size) {
    // Mock implementation
}

// PostgreSQL SPI Functions (mocks)
void* pg_table_open(const char* table_name) {
    return reinterpret_cast<void*>(0x4000);
}

int64_t pg_get_next_tuple(void* table_handle) {
    return 0;
}

int32_t pg_extract_field(void* tuple, int32_t field_index) {
    return 42;
}

void pg_store_result(void* result) {
    // Mock implementation
}

void pg_store_result_i32(int32_t value) {
    // Mock implementation
}

void pg_store_result_i64(int64_t value) {
    // Mock implementation
}

void pg_store_result_f64(double value) {
    // Mock implementation
}

void pg_store_result_text(const char* value) {
    // Mock implementation
}

} // extern "C"