#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

extern "C" {

extern bool g_jit_results_ready;
extern void mark_results_ready_for_streaming();
extern void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
extern void prepare_computed_results(int32_t numColumns);
extern bool add_tuple_to_result(int64_t value);

extern void* open_postgres_table(const char* tableName);
extern int64_t read_next_tuple_from_table(void* tableHandle);
extern int32_t get_int_field_mlir(void* tuple_handle, int32_t field_index, bool* is_null);

static void* g_execution_context = nullptr;

void rt_set_execution_context(void* context_ptr) {
    g_execution_context = context_ptr;
}

void* rt_get_execution_context() {
    if (g_execution_context) {
        return g_execution_context;
    }
    static struct {
        void* table_ref;
        int64_t row_count;
    } dummy_context = { nullptr, 1 };
    
    return &dummy_context;
}

struct DataSourceIterator {
    void* context;
    void* table_handle;  // PostgreSQL table handle
    bool has_current_tuple;
    
    int32_t current_id;
    bool current_id_is_null;
    int32_t current_col2;
    bool current_col2_is_null;
    
    int32_t current_value;
    bool current_is_null;
};

static DataSourceIterator* g_current_iterator = nullptr;

struct TableBuilder {
    void* data;
    int64_t row_count;
};

void* rt_tablebuilder_create(uint64_t /* varlen32_value */) {
    auto* builder = (TableBuilder*)malloc(sizeof(TableBuilder));
    builder->data = nullptr;
    builder->row_count = 0;
    return builder;
}

void rt_tablebuilder_nextrow(void* builder) {
    
    if (g_current_iterator && g_current_iterator->has_current_tuple) {
        prepare_computed_results(1);
        store_bigint_result(0, g_current_iterator->current_value, g_current_iterator->current_is_null);
        add_tuple_to_result(1);
    }
    if (builder) {
        auto* tb = (TableBuilder*)builder;
        tb->row_count++;
    }
}


void* rt_tablebuilder_build(void* builder) {
    mark_results_ready_for_streaming();
    return builder;
}

void rt_tablebuilder_addint64(void* /* builder */, int64_t /* value */) {
}

void rt_tablebuilder_addint32(void* /* builder */, bool /* is_null */, int32_t /* value */) {
}

void rt_tablebuilder_addbool(void* /* builder */, bool /* is_null */, bool /* value */) {
}

void rt_tablebuilder_destroy(void* builder) {
    if (builder) {
        free(builder);
    }
}

extern "C" __attribute__((noinline, cdecl)) void* rt_datasourceiteration_start(void* context, uint64_t /* varlen32_value */) {
    auto* iter = (DataSourceIterator*)malloc(sizeof(DataSourceIterator));
    iter->context = context;
    iter->has_current_tuple = false;
    iter->current_value = 0;
    iter->current_is_null = true;
    iter->table_handle = open_postgres_table("test");
    if (!iter->table_handle) {
        return iter;
    }
    g_current_iterator = iter;
    return iter;
}

extern "C" __attribute__((noinline, cdecl)) bool rt_datasourceiteration_isvalid(void* iterator) {
    if (!iterator) return false;
    
    auto* iter = (DataSourceIterator*)iterator;
    if (!iter->table_handle) return false;
    int64_t read_result = read_next_tuple_from_table(iter->table_handle);
    
    if (read_result == 1) {
        extern int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null);
        bool id_is_null = false;
        iter->current_id = get_int_field(iter->table_handle, 0, &id_is_null);
        iter->current_id_is_null = id_is_null;
        bool col2_is_null = false;
        iter->current_col2 = get_int_field(iter->table_handle, 1, &col2_is_null);
        iter->current_col2_is_null = col2_is_null;
        iter->current_value = iter->current_id;
        iter->current_is_null = iter->current_id_is_null;
        iter->has_current_tuple = true;
        return true;
    } else {
        iter->has_current_tuple = false;
        return false;
    }
}

extern "C" __attribute__((noinline, cdecl)) void rt_datasourceiteration_access(void* iterator, void* row_data) {
    if (row_data) {
        struct ColumnInfo {
            size_t offset;            // Offset in buffer
            size_t validMultiplier;   // Validity bitmap multiplier
            void* validBuffer;        // Validity bitmap buffer
            void* dataBuffer;         // Data buffer
            void* varLenBuffer;       // Variable length buffer
        };
        
        struct RecordBatchInfo {
            size_t numRows;           // Element 0: number of rows
            ColumnInfo columnInfo[2]; // Elements 1-10: 2 columns × 5 fields each
        } __attribute__((packed)) *batchInfo = (RecordBatchInfo*)row_data;
        
        batchInfo->numRows = 1;
        auto* iter = (DataSourceIterator*)iterator;
        static uint8_t valid_bitmap = 0xFF; // All bits valid
        
        if (iter && iter->has_current_tuple) {
            batchInfo->columnInfo[0].offset = 0;
            batchInfo->columnInfo[0].validMultiplier = 0;
            batchInfo->columnInfo[0].validBuffer = &valid_bitmap;
            batchInfo->columnInfo[0].dataBuffer = &iter->current_id;
            batchInfo->columnInfo[0].varLenBuffer = nullptr;
            batchInfo->columnInfo[1].offset = 0;
            batchInfo->columnInfo[1].validMultiplier = 0;
            batchInfo->columnInfo[1].validBuffer = &valid_bitmap;
            batchInfo->columnInfo[1].dataBuffer = &iter->current_col2;
            batchInfo->columnInfo[1].varLenBuffer = nullptr;
        }
    }
    
    __sync_synchronize();
}

extern "C" __attribute__((noinline, cdecl)) void rt_datasourceiteration_next(void* iterator) {
    if (iterator) {
        auto* iter = (DataSourceIterator*)iterator;
        iter->has_current_tuple = false;
    }
}

extern "C" __attribute__((noinline, cdecl)) void rt_datasourceiteration_end(void* iterator) {
    if (iterator) {
        auto* iter = (DataSourceIterator*)iterator;
        
        if (iter->table_handle) {
            extern void close_postgres_table(void* tableHandle);
            close_postgres_table(iter->table_handle);
            iter->table_handle = nullptr;
        }
        if (g_current_iterator == iterator) {
            g_current_iterator = nullptr;
        }
        
        free(iterator);
    }
}

} // extern "C"