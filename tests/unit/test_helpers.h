#pragma once

#include <vector>
#include <cstdint>

// Mock tuple scan context for testing
struct MockTupleScanContext {
    std::vector<int64_t> values;
    size_t currentIndex;
    bool hasMore;
};

extern MockTupleScanContext* g_mock_scan_context; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Mock runtime functions for unit tests
#ifndef POSTGRESQL_EXTENSION
extern "C" {
auto mock_get_next_tuple() -> int64_t;
auto open_postgres_table(int64_t tableName) -> int64_t;
auto read_next_tuple_from_table(int64_t tableHandle) -> int64_t;
auto close_postgres_table(int64_t tableHandle) -> void;
auto add_tuple_to_result(int64_t value) -> bool;
auto get_int_field(int64_t tuple_handle, int32_t field_index, bool* is_null) -> int32_t;
auto get_text_field(int64_t tuple_handle, int32_t field_index, bool* is_null) -> int64_t;
}
#endif