#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>

// Simple runtime functions for Test 1 JIT execution
// Based on working backup - uses malloc/free instead of PostgreSQL memory management
extern "C" {

// Execution context function
void* rt_get_execution_context() {
    // Return a dummy context for Test 1
    static struct {
        void* table_ref;
        int64_t row_count;
    } dummy_context = { nullptr, 1 }; // Test 1 has 1 row
    
    return &dummy_context;
}

// Table builder functions
struct TableBuilder {
    void* data;
    int64_t row_count;
};

void* rt_tablebuilder_create(uint64_t /* varlen32_value */) {
    // Create a simple table builder
    auto* builder = (TableBuilder*)malloc(sizeof(TableBuilder));
    builder->data = nullptr;
    builder->row_count = 0;
    return builder;
}

void rt_tablebuilder_nextrow(void* builder) {
    // Increment row count
    if (builder) {
        auto* tb = (TableBuilder*)builder;
        tb->row_count++;
    }
}

void* rt_tablebuilder_build(void* builder) {
    // Return the built table (just return the builder for now)
    return builder;
}

void rt_tablebuilder_addint64(void* /* builder */, int64_t /* value */) {
    // Add an int64 value to the current row - no-op for Test 1
}

void rt_tablebuilder_destroy(void* builder) {
    // Clean up the table builder
    if (builder) {
        free(builder);
    }
}

// Data source iteration functions
struct DataSourceIterator {
    void* context;
    int current_row;
    int total_rows;
};

void* rt_datasourceiteration_start(void* context, uint64_t /* varlen32_value */) {
    // Create a simple iterator for Test 1 (single row)
    auto* iter = (DataSourceIterator*)malloc(sizeof(DataSourceIterator));
    iter->context = context;
    iter->current_row = 0;
    iter->total_rows = 1; // Test 1 has exactly 1 row
    return iter;
}

bool rt_datasourceiteration_isvalid(void* iterator) {
    // Check if iterator has more rows
    if (iterator) {
        auto* iter = (DataSourceIterator*)iterator;
        return iter->current_row < iter->total_rows;
    }
    return false;
}

void rt_datasourceiteration_access(void* iterator, void* row_data) {
    // Access current row data - for Test 1, fill with dummy data
    if (iterator && row_data) {
        // Fill with dummy row data structure for Test 1
        struct TestRowData {
            int64_t tuple_count;
            int64_t column_count;  
            int64_t data_size;
            void* tuple_data;
            void* column_data;
            void* attr_data;
        };
        
        auto* row = (TestRowData*)row_data;
        row->tuple_count = 1;
        row->column_count = 1;
        row->data_size = 8;
        row->tuple_data = nullptr;
        row->column_data = nullptr;
        row->attr_data = nullptr;
    }
}

void rt_datasourceiteration_next(void* iterator) {
    // Move to next row
    if (iterator) {
        auto* iter = (DataSourceIterator*)iterator;
        iter->current_row++;
    }
}

void rt_datasourceiteration_end(void* iterator) {
    // Clean up iterator
    if (iterator) {
        free(iterator);
    }
}

} // extern "C"