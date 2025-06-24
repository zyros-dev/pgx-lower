#pragma once

#include <vector>
#include <cstdint>

// Mock tuple scan context for testing
struct MockTupleScanContext {
    std::vector<int64_t> values;
    size_t currentIndex;
    bool hasMore;
};

extern MockTupleScanContext* g_mock_scan_context;

// Mock runtime functions for unit tests
#ifndef POSTGRESQL_EXTENSION
extern "C" {
    int64_t mock_get_next_tuple();
    int64_t open_postgres_table(int64_t tableName);
    int64_t read_next_tuple_from_table(int64_t tableHandle);
    void close_postgres_table(int64_t tableHandle);
    bool add_tuple_to_result(int64_t value);
    int32_t get_int_field(int64_t tuple_handle, int32_t field_index, bool* is_null);
    int64_t get_text_field(int64_t tuple_handle, int32_t field_index, bool* is_null);
}
#endif