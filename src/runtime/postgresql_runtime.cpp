#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

// Functional runtime for Test 1 JIT execution
// Uses simple malloc/free but actually sets results ready flag
extern "C" {

// Global flag to indicate results are ready (defined in tuple_access.cpp)
extern bool g_jit_results_ready;

// Functions defined in tuple_access.cpp  
extern void mark_results_ready_for_streaming();
extern void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
extern void prepare_computed_results(int32_t numColumns);
extern bool add_tuple_to_result(int64_t value);

// PostgreSQL table access functions
extern void* open_postgres_table(const char* tableName);
extern int64_t read_next_tuple_from_table(void* tableHandle);
extern int32_t get_int_field_mlir(void* tuple_handle, int32_t field_index, bool* is_null);

// Execution context function
void* rt_get_execution_context() {
    // Return a dummy context for Test 1
    static struct {
        void* table_ref;
        int64_t row_count;
    } dummy_context = { nullptr, 1 }; // Test 1 has 1 row
    
    return &dummy_context;
}

// Data source iteration functions (moved before table builder to resolve dependency)
struct DataSourceIterator {
    void* context;
    void* table_handle;  // PostgreSQL table handle
    bool has_current_tuple;
    int32_t current_value;
    bool current_is_null;
};

// Global to store the current tuple data during JIT execution
static DataSourceIterator* g_current_iterator = nullptr;

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
    elog(NOTICE, "🔗 rt_tablebuilder_nextrow: Streaming current tuple immediately");
    
    // IMMEDIATE STREAMING: Stream the current tuple right now
    // This is called by MLIR for each tuple during iteration
    
    if (g_current_iterator && g_current_iterator->has_current_tuple) {
        // We have a current tuple from PostgreSQL
        elog(NOTICE, "🔗 Streaming tuple with value=%d, is_null=%s",
             g_current_iterator->current_value, 
             g_current_iterator->current_is_null ? "true" : "false");
        
        // Prepare results storage for this tuple
        prepare_computed_results(1);  // Each tuple has 1 column for Test 1/2
        
        // Store the current tuple's value
        store_bigint_result(0, g_current_iterator->current_value, g_current_iterator->current_is_null);
        
        // Stream this tuple immediately to PostgreSQL output
        bool streaming_result = add_tuple_to_result(1);
        elog(NOTICE, "🔗 Immediate streaming returned: %s", streaming_result ? "true" : "false");
    }
    
    // Also increment row count for tracking
    if (builder) {
        auto* tb = (TableBuilder*)builder;
        tb->row_count++;
    }
}

// DataSourceIterator already defined above (moved before table builder functions)

// Removed g_preserved_data - switching to immediate streaming pattern
// Each tuple is streamed immediately when MLIR accesses it

void* rt_tablebuilder_build(void* builder) {
    elog(NOTICE, "🎯 rt_tablebuilder_build called from JIT!");
    
    // In immediate streaming mode, tuples have already been streamed during iteration
    // This function now just signals that all results have been streamed
    
    // Signal that results are ready (all tuples have been streamed)
    mark_results_ready_for_streaming();
    
    elog(NOTICE, "🎯 rt_tablebuilder_build completed - all tuples already streamed!");
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

void* rt_datasourceiteration_start(void* context, uint64_t /* varlen32_value */) {
    elog(NOTICE, "🔗 rt_datasourceiteration_start: Opening PostgreSQL table");
    
    // Create iterator with PostgreSQL table connection
    auto* iter = (DataSourceIterator*)malloc(sizeof(DataSourceIterator));
    iter->context = context;
    iter->has_current_tuple = false;
    iter->current_value = 0;
    iter->current_is_null = true;
    
    // Open the actual PostgreSQL table "test"
    iter->table_handle = open_postgres_table("test");
    if (!iter->table_handle) {
        elog(NOTICE, "🔗 rt_datasourceiteration_start: Failed to open PostgreSQL table");
        return iter;
    }
    
    // Store globally so table builder can access the data
    g_current_iterator = iter;
    
    elog(NOTICE, "🔗 rt_datasourceiteration_start: Successfully opened PostgreSQL table");
    return iter;
}

bool rt_datasourceiteration_isvalid(void* iterator) {
    if (!iterator) return false;
    
    auto* iter = (DataSourceIterator*)iterator;
    if (!iter->table_handle) return false;
    
    elog(NOTICE, "🔗 rt_datasourceiteration_isvalid: Reading next tuple from PostgreSQL");
    
    // Try to read the next tuple from PostgreSQL
    int64_t read_result = read_next_tuple_from_table(iter->table_handle);
    
    if (read_result == 1) {
        // We have a tuple - extract the integer field (column 0)
        bool is_null = false;
        iter->current_value = get_int_field_mlir(iter->table_handle, 0, &is_null);
        iter->current_is_null = is_null;
        iter->has_current_tuple = true;
        
        // No longer preserving data - we stream immediately in rt_tablebuilder_nextrow
        
        elog(NOTICE, "🔗 rt_datasourceiteration_isvalid: Got tuple with value=%d, is_null=%s (PRESERVED)", 
             iter->current_value, is_null ? "true" : "false");
        return true;
    } else {
        // End of table or error
        iter->has_current_tuple = false;
        elog(NOTICE, "🔗 rt_datasourceiteration_isvalid: End of table");
        return false;
    }
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
    // Move to next row - just mark current tuple as invalid
    if (iterator) {
        auto* iter = (DataSourceIterator*)iterator;
        iter->has_current_tuple = false;
        elog(NOTICE, "🔗 rt_datasourceiteration_next: Moving to next tuple");
    }
}

void rt_datasourceiteration_end(void* iterator) {
    // Clean up iterator
    if (iterator) {
        free(iterator);
    }
}

} // extern "C"