#include "test_helpers.h"

MockTupleScanContext* g_mock_scan_context = nullptr;

extern "C" auto mock_get_next_tuple() -> int64_t {
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

extern "C" auto open_postgres_table(int64_t /*tableName*/) -> int64_t {
    if (!g_mock_scan_context) {
        return 0;
    }
    return reinterpret_cast<int64_t>(g_mock_scan_context);
}

extern "C" auto read_next_tuple_from_table(int64_t tableHandle) -> int64_t {
    if (!tableHandle) {
        return -1;
    }

    auto* context = reinterpret_cast<MockTupleScanContext*>(tableHandle);
    (void)context; // Suppress unused variable warning
    return mock_get_next_tuple();
}

extern "C" auto close_postgres_table(int64_t /*tableHandle*/) -> void {
    // Nothing to do for mock implementation
}

extern "C" auto add_tuple_to_result(int64_t /*value*/) -> bool {
    return true;
}

extern "C" auto get_int_field(int64_t /*tuple_handle*/, int32_t field_index, bool* is_null) -> int32_t {
    // Mock implementation for unit tests
    *is_null = false;
    return field_index * 42; // Return predictable values
}

extern "C" auto get_text_field(int64_t /*tuple_handle*/, int32_t /*field_index*/, bool* is_null) -> int64_t {
    // Mock implementation for unit tests
    static const char* mock_text = "mock_text_field";
    *is_null = false;
    return reinterpret_cast<int64_t>(mock_text);
}

extern "C" auto get_numeric_field(void* /*tuple_handle*/, int32_t field_index, bool* is_null) -> double {
    // Mock implementation for unit tests
    *is_null = false;
    return static_cast<double>(field_index * 3.14); // Return predictable values
}

// Mock implementations for result storage functions
extern "C" void store_int_result(int32_t /*columnIndex*/, int32_t /*value*/, bool /*isNull*/) {
    // Mock implementation for unit tests - just return
}

extern "C" void store_bool_result(int32_t /*columnIndex*/, bool /*value*/, bool /*isNull*/) {
    // Mock implementation for unit tests - just return
}

extern "C" void store_bigint_result(int32_t /*columnIndex*/, int64_t /*value*/, bool /*isNull*/) {
    // Mock implementation for unit tests - just return
}

extern "C" void store_text_result(int32_t /*columnIndex*/, const char* /*value*/, bool /*isNull*/) {
    // Mock implementation for unit tests - just return
}

extern "C" void prepare_computed_results(int32_t /*numColumns*/) {
    // Mock implementation for unit tests - just return
}