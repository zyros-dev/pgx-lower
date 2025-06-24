#include "test_helpers.h"

MockTupleScanContext* g_mock_scan_context = nullptr;

#ifndef POSTGRESQL_EXTENSION
extern "C" int64_t mock_get_next_tuple() {
    if (!g_mock_scan_context) {
        return -1;
    }
    
    if (g_mock_scan_context->currentIndex >= g_mock_scan_context->values.size()) {
        g_mock_scan_context->hasMore = false;
        return -2;
    }
    
    int64_t value = g_mock_scan_context->values[g_mock_scan_context->currentIndex];
    g_mock_scan_context->currentIndex++;
    g_mock_scan_context->hasMore = true;
    
    return value;
}

extern "C" int64_t open_postgres_table(int64_t tableName) {
    if (!g_mock_scan_context) {
        return 0;
    }
    return reinterpret_cast<int64_t>(g_mock_scan_context);
}

extern "C" int64_t read_next_tuple_from_table(int64_t tableHandle) {
    if (!tableHandle) {
        return -1;
    }
    
    MockTupleScanContext* context = reinterpret_cast<MockTupleScanContext*>(tableHandle);
    return mock_get_next_tuple();
}

extern "C" void close_postgres_table(int64_t tableHandle) {
    // Nothing to do for mock implementation
}

extern "C" bool add_tuple_to_result(int64_t value) {
    return true;
}

extern "C" int32_t get_int_field(int64_t tuple_handle, int32_t field_index, bool* is_null) {
    // Mock implementation for unit tests
    *is_null = false;
    return field_index * 42; // Return predictable values
}

extern "C" int64_t get_text_field(int64_t tuple_handle, int32_t field_index, bool* is_null) {
    // Mock implementation for unit tests
    static const char* mock_text = "mock_text_field";
    *is_null = false;
    return reinterpret_cast<int64_t>(mock_text);
}
#endif