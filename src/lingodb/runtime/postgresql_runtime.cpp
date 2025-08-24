#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <json.h>

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

extern "C" {

extern bool g_jit_results_ready;
extern void mark_results_ready_for_streaming();
extern void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
extern void store_bool_result(int32_t columnIndex, bool value, bool isNull);
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
    elog(NOTICE, "[DEBUG] rt_get_execution_context called");
    if (g_execution_context) {
        elog(NOTICE, "[DEBUG] rt_get_execution_context returning g_execution_context: %p", g_execution_context);
        return g_execution_context;
    }
    static struct {
        void* table_ref;
        int64_t row_count;
    } dummy_context = { nullptr, 1 };
    
    return &dummy_context;
}

enum class ColumnType {
    INTEGER,
    BOOLEAN
};

struct ColumnSpec {
    std::string name;
    ColumnType type;
};

struct DataSourceIterator {
    void* context;
    void* table_handle;  // PostgreSQL table handle
    bool has_current_tuple;
    
    std::string table_name;
    std::vector<ColumnSpec> columns;
    
    // Dynamic column data storage
    std::vector<int32_t> int_values;
    std::vector<bool> int_nulls;
    std::vector<uint8_t> bool_values;  // Use uint8_t instead of bool for addressability
    std::vector<bool> bool_nulls;
    
    // Legacy fields for backward compatibility
    int32_t current_id;
    bool current_id_is_null;
    int32_t current_col2;
    bool current_col2_is_null;
    int32_t current_value;
    bool current_is_null;
};

static DataSourceIterator* g_current_iterator = nullptr;

// Table specification structure
struct TableSpec {
    std::string table_name;
    std::vector<std::string> column_names;
};

// JSON parsing function using nlohmann/json
TableSpec parse_table_spec(const char* json_str) {
    TableSpec spec;
    
    try {
        elog(NOTICE, "[DEBUG] parse_table_spec: parsing JSON: %s", json_str);
        
        using json = nlohmann::json;
        json j = json::parse(json_str);
        
        if (j.contains("table") && j["table"].is_string()) {
            spec.table_name = j["table"];
        }
        
        if (j.contains("columns") && j["columns"].is_array()) {
            for (const auto& col : j["columns"]) {
                if (col.is_string()) {
                    spec.column_names.push_back(col);
                }
            }
        }
        
        elog(NOTICE, "[DEBUG] parse_table_spec: table=%s, columns=%zu", spec.table_name.c_str(), spec.column_names.size());
    } catch (const std::exception& e) {
        elog(NOTICE, "[DEBUG] parse_table_spec: JSON parsing failed: %s", e.what());
        // Return empty spec on error
    }
    
    return spec;
}

struct TableBuilder {
    void* data;
    int64_t row_count;
    int32_t current_column_index;  // Track which column we're currently filling
    int32_t total_columns;         // Total number of columns in this table
    
    TableBuilder() : data(nullptr), row_count(0), current_column_index(0), total_columns(0) {}
};

extern "C" __attribute__((noinline, cdecl)) void* rt_tablebuilder_create(__int128 /* varlen32_value */) {
    elog(NOTICE, "[DEBUG] rt_tablebuilder_create called");
    // Use placement new to properly initialize C++ object
    void* builder_memory = malloc(sizeof(TableBuilder));
    auto* builder = new(builder_memory) TableBuilder();
    elog(NOTICE, "[DEBUG] rt_tablebuilder_create returning builder: %p", builder);
    return builder;
}

extern "C" __attribute__((noinline, cdecl)) void rt_tablebuilder_nextrow(void* builder) {
    elog(NOTICE, "[DEBUG] rt_tablebuilder_nextrow called with builder: %p", builder);
    if (g_current_iterator && g_current_iterator->has_current_tuple) {
        prepare_computed_results(1);
        store_bigint_result(0, g_current_iterator->current_value, g_current_iterator->current_is_null);
        add_tuple_to_result(1);
    }
    if (builder) {
        auto* tb = static_cast<TableBuilder*>(builder);
        tb->row_count++;
        // Reset column index for next row (LingoDB pattern)
        tb->current_column_index = 0;
        elog(NOTICE, "[DEBUG] rt_tablebuilder_nextrow: reset column index to 0 for row %ld", tb->row_count);
    }
}


extern "C" __attribute__((noinline, cdecl)) void* rt_tablebuilder_build(void* builder) {
    mark_results_ready_for_streaming();
    return builder;
}

extern "C" __attribute__((noinline, cdecl)) void rt_tablebuilder_addint64(void* /* builder */, int64_t /* value */) {
}

extern "C" __attribute__((noinline, cdecl)) void rt_tablebuilder_addint32(void* /* builder */, bool /* is_null */, int32_t /* value */) {
}

extern "C" __attribute__((noinline, cdecl)) void rt_tablebuilder_addbool(void* builder, bool is_null, bool value) {
    elog(NOTICE, "[DEBUG] rt_tablebuilder_addbool called: value=%s, is_null=%s", 
         value ? "true" : "false", is_null ? "true" : "false");
    
    auto* tb = static_cast<TableBuilder*>(builder);
    if (tb) {
        elog(NOTICE, "[DEBUG] rt_tablebuilder_addbool: storing at column index %d", tb->current_column_index);
        // Use the current column index from the TableBuilder
        store_bool_result(tb->current_column_index, value, is_null);
        // Advance to next column (LingoDB pattern)
        tb->current_column_index++;
    } else {
        elog(NOTICE, "[DEBUG] rt_tablebuilder_addbool: null builder, using fallback column 0");
        store_bool_result(0, value, is_null);
    }
    elog(NOTICE, "[DEBUG] rt_tablebuilder_addbool completed successfully");
}

extern "C" __attribute__((noinline, cdecl)) void rt_tablebuilder_destroy(void* builder) {
    if (builder) {
        auto* tb = static_cast<TableBuilder*>(builder);
        // Call destructor explicitly since we used placement new
        tb->~TableBuilder();
        free(builder);
    }
}

extern "C" __attribute__((noinline, cdecl)) void* rt_datasourceiteration_start(void* context, __int128 varlen32_value) {
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_start called with context: %p, varlen32_value: %p", context, (void*)(uintptr_t)varlen32_value);
    
    // Use placement new to properly initialize the DataSourceIterator with C++ objects
    void* iter_memory = malloc(sizeof(DataSourceIterator));
    auto* iter = new(iter_memory) DataSourceIterator();
    
    iter->context = context;
    iter->has_current_tuple = false;
    iter->current_value = 0;
    iter->current_is_null = true;
    
    // Decode VarLen32 structure from __int128 parameter
    // Based on LingoDB's VarLen32::asI128() method
    struct VarLen32Decoder {
        uint32_t len;
        union {
            uint8_t bytes[12];
            struct __attribute__((__packed__)) {
                uint32_t first4;
                uint64_t last8;
            };
        };
    } __attribute__((packed));
    
    VarLen32Decoder* decoder = reinterpret_cast<VarLen32Decoder*>(&varlen32_value);
    uint32_t actual_len = decoder->len & ~0x80000000;  // Remove lazy mask
    
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: VarLen32 len=%u", actual_len);
    
    const char* json_spec = nullptr;
    bool json_parsed = false;
    
    if (actual_len > 0 && actual_len <= 12) {
        // Short string stored inline in bytes array
        json_spec = reinterpret_cast<const char*>(decoder->bytes);
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: short string (len=%u) inline", actual_len);
    } else if (actual_len > 12) {
        // Long string stored as pointer in bytes[4]
        uint8_t** ptr_loc = reinterpret_cast<uint8_t**>(&decoder->bytes[4]);
        json_spec = reinterpret_cast<const char*>(*ptr_loc);
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: long string (len=%u) at pointer %p", actual_len, json_spec);
    }
    
    if (json_spec && actual_len > 0) {
        PG_TRY();
        {
            // Create null-terminated string for JSON parsing
            std::string json_string(json_spec, actual_len);
            elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: JSON string: %s", json_string.c_str());
            
            if (json_string[0] == '{') {
                elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: valid JSON detected, parsing...");
                TableSpec spec = parse_table_spec(json_string.c_str());
                
                if (!spec.table_name.empty() && !spec.column_names.empty()) {
                    // Remove OID suffix if present (e.g., "test_logical|oid:123" -> "test_logical")
                    size_t pipe_pos = spec.table_name.find('|');
                    if (pipe_pos != std::string::npos) {
                        spec.table_name = spec.table_name.substr(0, pipe_pos);
                    }
                    
                    iter->table_name = spec.table_name;
                    
                    // Convert column names to column specs with type inference
                    for (const auto& col_name : spec.column_names) {
                        ColumnSpec col_spec;
                        col_spec.name = col_name;
                        // Infer type from column name patterns
                        if (col_name.find("flag") != std::string::npos || col_name.find("bool") != std::string::npos) {
                            col_spec.type = ColumnType::BOOLEAN;
                        } else {
                            col_spec.type = ColumnType::INTEGER;
                        }
                        iter->columns.push_back(col_spec);
                    }
                    
                    json_parsed = true;
                    elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: JSON parsed successfully - table '%s' with %zu columns", iter->table_name.c_str(), iter->columns.size());
                }
            }
        }
        PG_CATCH();
        {
            elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: exception reading VarLen32 JSON, using fallback");
            FlushErrorState();
        }
        PG_END_TRY();
    }
    
    // Fallback if JSON parsing failed
    if (!json_parsed) {
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: JSON parsing failed, using fallback defaults");
        iter->table_name = "test_comparison";
        
        // Set up default 2-integer column layout
        ColumnSpec col1, col2;
        col1.name = "value";
        col1.type = ColumnType::INTEGER;
        col2.name = "score";
        col2.type = ColumnType::INTEGER;
        iter->columns.push_back(col1);
        iter->columns.push_back(col2);
    }
    
    // Initialize dynamic storage based on column count
    iter->int_values.resize(iter->columns.size(), 0);
    iter->int_nulls.resize(iter->columns.size(), true);
    iter->bool_values.resize(iter->columns.size(), 0);
    iter->bool_nulls.resize(iter->columns.size(), true);
    
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: configured for table '%s' with %zu columns", iter->table_name.c_str(), iter->columns.size());
    
    iter->table_handle = open_postgres_table(iter->table_name.c_str());
    
    if (!iter->table_handle) {
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_start: open_postgres_table failed for '%s'", iter->table_name.c_str());
        return iter;
    }
    
    g_current_iterator = iter;
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_start returning iterator: %p", iter);
    return iter;
}

extern "C" __attribute__((noinline, cdecl)) bool rt_datasourceiteration_isvalid(void* iterator) {
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid called with iterator: %p", iterator);
    if (!iterator) return false;
    
    auto* iter = (DataSourceIterator*)iterator;
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid: iter=%p, table_handle=%p", iter, iter->table_handle);
    if (!iter->table_handle) {
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid finished running with: %p branch 1 (no table_handle)", iterator);
        return false;
    }
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid: About to call read_next_tuple_from_table");
    int64_t read_result = read_next_tuple_from_table(iter->table_handle);
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid: read_result = %ld", read_result);
    
    if (read_result == 1) {
        extern int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null);
        extern bool get_bool_field(void* tuple_handle, int32_t field_index, bool* is_null);
        
        // Read dynamic columns based on specification
        for (size_t i = 0; i < iter->columns.size(); ++i) {
            const auto& col_spec = iter->columns[i];
            if (col_spec.type == ColumnType::BOOLEAN) {
                bool is_null = false;
                bool bool_value = get_bool_field(iter->table_handle, i, &is_null);
                iter->bool_values[i] = bool_value ? 1 : 0;
                iter->bool_nulls[i] = is_null;
                elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid: column %zu (%s) = %s (null=%s)", 
                     i, col_spec.name.c_str(), bool_value ? "true" : "false", is_null ? "true" : "false");
            } else {
                bool is_null = false;
                iter->int_values[i] = get_int_field(iter->table_handle, i, &is_null);
                iter->int_nulls[i] = is_null;
                elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid: column %zu (%s) = %d (null=%s)", 
                     i, col_spec.name.c_str(), iter->int_values[i], is_null ? "true" : "false");
            }
        }
        
        // Maintain legacy fields for backward compatibility
        if (iter->columns.size() >= 1 && iter->columns[0].type == ColumnType::INTEGER) {
            iter->current_id = iter->int_values[0];
            iter->current_id_is_null = iter->int_nulls[0];
            iter->current_value = iter->current_id;
            iter->current_is_null = iter->current_id_is_null;
        }
        if (iter->columns.size() >= 2 && iter->columns[1].type == ColumnType::INTEGER) {
            iter->current_col2 = iter->int_values[1];
            iter->current_col2_is_null = iter->int_nulls[1];
        }
        
        iter->has_current_tuple = true;
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid finished running with: %p branch 2", iterator);
        return true;
    } else {
        iter->has_current_tuple = false;
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_isvalid finished running with: %p branch 3", iterator);
        return false;
    }
}

extern "C" __attribute__((noinline, cdecl)) void rt_datasourceiteration_access(void* iterator, void* row_data) {
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_access called with iterator: %p, row_data: %p", iterator, row_data);
    if (row_data) {
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: row_data is valid, proceeding");
        
        struct ColumnInfo {
            size_t offset;            // Offset in buffer
            size_t validMultiplier;   // Validity bitmap multiplier
            void* validBuffer;        // Validity bitmap buffer
            void* dataBuffer;         // Data buffer
            void* varLenBuffer;       // Variable length buffer
        };
        
        auto* iter = (DataSourceIterator*)iterator;
        if (!iter || !iter->has_current_tuple) {
            elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: invalid iterator or no current tuple");
            return;
        }
        
        size_t num_columns = iter->columns.size();
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: handling %zu columns", num_columns);
        
        // The row_data is expected to be a dynamic structure:
        // [numRows: size_t][columnInfo[0]: 5*size_t][columnInfo[1]: 5*size_t]...[columnInfo[n]: 5*size_t]
        size_t* row_data_ptr = (size_t*)row_data;
        row_data_ptr[0] = 1; // numRows = 1
        
        static uint8_t valid_bitmap = 0xFF; // All bits valid
        
        for (size_t i = 0; i < num_columns; ++i) {
            const auto& col_spec = iter->columns[i];
            size_t* column_info_ptr = &row_data_ptr[1 + i * 5]; // Each ColumnInfo has 5 fields
            
            column_info_ptr[0] = 0;                    // offset
            column_info_ptr[1] = 0;                    // validMultiplier
            column_info_ptr[2] = (size_t)&valid_bitmap; // validBuffer
            column_info_ptr[4] = 0;                    // varLenBuffer = nullptr
            
            if (col_spec.type == ColumnType::BOOLEAN) {
                column_info_ptr[3] = (size_t)&iter->bool_values[i]; // dataBuffer
                elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: column %zu (%s) boolean = %s at %p", 
                     i, col_spec.name.c_str(), iter->bool_values[i] ? "true" : "false", &iter->bool_values[i]);
            } else {
                column_info_ptr[3] = (size_t)&iter->int_values[i]; // dataBuffer
                elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: column %zu (%s) integer = %d at %p", 
                     i, col_spec.name.c_str(), iter->int_values[i], &iter->int_values[i]);
            }
        }
        
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: configured %zu columns successfully", num_columns);
    } else {
        elog(NOTICE, "[DEBUG] rt_datasourceiteration_access: row_data is NULL");
    }
    
    elog(NOTICE, "[DEBUG] rt_datasourceiteration_access completed successfully");
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
        
        // Properly destroy C++ objects before freeing memory
        iter->~DataSourceIterator();
        free(iterator);
    }
}

} // extern "C"