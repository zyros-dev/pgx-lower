#include "pgx-lower/runtime/PostgreSQLRuntime.h"
#include "lingodb/runtime/DataSourceIteration.h"
#include "pgx-lower/runtime/StringRuntime.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>
#include <json.h>
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/runtime/runtime_templates.h"
#include "pgx-lower/utility/logging.h"

// Need access to g_computed_results for decimal handling
extern ComputedResultStorage g_computed_results;

extern "C" {
#include "postgres.h"
#include "access/htup_details.h"
#include "catalog/pg_type_d.h"
#include "utils/elog.h"
#include "utils/numeric.h"
#include "fmgr.h"
}

extern "C" {
extern void mark_results_ready_for_streaming();
extern void store_bigint_result(int32_t columnIndex, int64_t value, bool isNull);
extern void store_bool_result(int32_t columnIndex, bool value, bool isNull);
extern void prepare_computed_results(int32_t numColumns);
extern bool add_tuple_to_result(int64_t value);
extern void* open_postgres_table(const char* tableName);
extern int64_t read_next_tuple_from_table(void* tableHandle);
}

// ============================================================================
// Internal C implementation functions
// ============================================================================

static void* g_execution_context = nullptr;

extern "C" {

// Keep the original names that JIT expects
void rt_set_execution_context(void* context_ptr) {
    g_execution_context = context_ptr;
}

void* rt_get_execution_context() {
    PGX_LOG(RUNTIME, IO, "rt_get_execution_context IN");
    PGX_LOG(RUNTIME, DEBUG, "rt_get_execution_context called");
    if (g_execution_context) {
        PGX_LOG(RUNTIME, DEBUG, "rt_get_execution_context returning g_execution_context: %p", g_execution_context);
        PGX_LOG(RUNTIME, IO, "rt_get_execution_context OUT: %p", g_execution_context);
        return g_execution_context;
    }
    static struct {
        void* table_ref;
        int64_t row_count;
    } dummy_context = { nullptr, 1 };

    PGX_LOG(RUNTIME, IO, "rt_get_execution_context OUT: %p (dummy)", &dummy_context);
    return &dummy_context;
}

// ColumnType enum - keep outside namespace runtime for internal use
enum class ColumnType {
    INTEGER,
    BIGINT,
    BOOLEAN,
    STRING,
    TEXT,
    VARCHAR
};

struct ColumnSpec {
    std::string name;
    ColumnType type;
};

struct DataSourceIterator {
    void* context;
    void* table_handle;
    bool has_current_tuple;

    std::string table_name;
    std::vector<ColumnSpec> columns;

    // Dynamic column data storage
    std::vector<int32_t> int_values;
    std::vector<bool> int_nulls;
    std::vector<int64_t> bigint_values;
    std::vector<bool> bigint_nulls;
    std::vector<uint8_t> bool_values;  // Use uint8_t instead of bool for addressability
    std::vector<bool> bool_nulls;
    
    // Per-column string storage with accumulation for operations like ORDER BY
    std::vector<std::vector<int32_t>> string_offsets_per_column;
    std::vector<std::vector<uint8_t>> string_data_buffers_per_column;
    std::vector<bool> string_nulls;
    std::vector<std::string> string_data;
    
    int32_t current_row_index = 0;

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

static int get_column_position(const std::string& table_name, const std::string& column_name) {
    extern int32_t get_column_attnum(const char* table_name, const char* column_name);
    int32_t attnum = get_column_attnum(table_name.c_str(), column_name.c_str());
    if (attnum > 0) {
        // Convert from 1-based PostgreSQL attnum to 0-based index
        return attnum - 1;
    }

    return -1;  // Column not found
}

static TableSpec parse_table_spec(const char* json_str) {
    PGX_LOG(RUNTIME, IO, "parse_table_spec IN: json_str=%s", json_str ? json_str : "NULL");
    TableSpec spec;

    try {
        PGX_LOG(RUNTIME, DEBUG, "parse_table_spec: parsing JSON: %s", json_str);

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

        PGX_LOG(RUNTIME, DEBUG, "parse_table_spec: table=%s, columns=%zu", spec.table_name.c_str(), spec.column_names.size());
    } catch (const std::exception& e) {
        PGX_LOG(RUNTIME, DEBUG, "parse_table_spec: JSON parsing failed: %s", e.what());
        // Return empty spec on error
    }

    PGX_LOG(RUNTIME, IO, "parse_table_spec OUT: table=%s, columns=%zu", spec.table_name.c_str(), spec.column_names.size());
    return spec;
}

// Memory context callback for TableBuilder cleanup
static void cleanup_tablebuilder_callback(void* arg) {
    runtime::TableBuilder* tb = static_cast<runtime::TableBuilder*>(arg);
}

namespace runtime {

// Constructor
TableBuilder::TableBuilder() 
    : data(nullptr), row_count(0), current_column_index(0), total_columns(0) {
}

// Static factory method
TableBuilder* TableBuilder::create(VarLen32 schema_param) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_create IN: schema_param len=%u", schema_param.getLen());

    MemoryContext oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    void* builder_memory = palloc(sizeof(runtime::TableBuilder));
    runtime::TableBuilder* builder = new(builder_memory) runtime::TableBuilder();

    builder->total_columns = 0;
    PGX_LOG(RUNTIME, DEBUG, "rt_tablebuilder_create: initialized with dynamic column tracking");

    MemoryContextCallback* callback = (MemoryContextCallback*)palloc(sizeof(MemoryContextCallback));
    callback->func = cleanup_tablebuilder_callback;
    callback->arg = builder;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

    MemoryContextSwitchTo(oldcontext);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_create OUT: builder=%p, total_columns=%d", builder, builder->total_columns);
    return builder;
}

void TableBuilder::nextRow() {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_nextrow IN: builder=%p", this);

    // 'this' is the TableBuilder instance

    // LingoDB currColumn assertion pattern: verify all columns were filled
    if (current_column_index != total_columns) {
        PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: column count info - expected %d columns, got %d (this may be normal during MLIR pipeline development)",
             total_columns, current_column_index);
    } else {
        PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: LingoDB column validation passed - %d columns filled", current_column_index);
    }

    row_count++;

    if (total_columns > 0) {
        PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: submitting row with %d columns", total_columns);
        add_tuple_to_result(total_columns);
    }

    current_column_index = 0;
    PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: reset column index to 0 for row %ld", row_count);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_nextrow OUT");
}

TableBuilder* TableBuilder::build() {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_build IN: builder=%p", this);
    

    mark_results_ready_for_streaming();

    PGX_LOG(RUNTIME, DEBUG, "TableBuilder state before return:");
    PGX_LOG(RUNTIME, DEBUG, "\t- builder address: %p", this);
    PGX_LOG(RUNTIME, DEBUG, "\t- row_count: %ld", row_count);
    PGX_LOG(RUNTIME, DEBUG, "\t- total_columns: %d", total_columns);
    PGX_LOG(RUNTIME, DEBUG, "\t- current_column_index: %d", current_column_index);
    if (g_computed_results.numComputedColumns > 0) {
        PGX_LOG(RUNTIME, DEBUG, "\t- computed columns: %d", g_computed_results.numComputedColumns);
        for (int i = 0; i < g_computed_results.numComputedColumns && i < 10; i++) {
            PGX_LOG(RUNTIME, DEBUG, "\t\t- col[%d]: type=%d, null=%d",
                    i, g_computed_results.computedTypes[i], g_computed_results.computedNulls[i]);
        }
    }

    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_build OUT: returning builder=%p", this);
    return this;
}

void TableBuilder::addInt64(bool is_valid, int64_t value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint64 IN: value=%ld, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint64 OUT");
}

void TableBuilder::addInt32(bool is_valid, int32_t value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint32 IN: value=%d, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<int32_t>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint32 OUT");
}

void TableBuilder::addBool(bool is_valid, bool value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addbool IN: value=%s, is_valid=%s", value ? "true" : "false", is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<bool>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addbool OUT");
}

void TableBuilder::addInt8(bool is_valid, int8_t value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint8 IN: value=%d, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<int8_t>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint8 OUT");
}

void TableBuilder::addInt16(bool is_valid, int16_t value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint16 IN: value=%d, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<int16_t>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addint16 OUT");
}

void TableBuilder::addFloat32(bool is_valid, float value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addfloat32 IN: value=%f, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<float>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addfloat32 OUT");
}

void TableBuilder::addFloat64(bool is_valid, double value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addfloat64 IN: value=%f, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<double>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addfloat64 OUT");
}

void TableBuilder::addBinary(bool is_valid, VarLen32 value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addbinary IN: len=%u, is_valid=%s", value.getLen(), is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<runtime::VarLen32>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addbinary OUT");
}

void TableBuilder::addDecimal(bool is_valid, __int128 value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_adddecimal IN: is_valid=%s", is_valid ? "true" : "false");
    // Just truncate to int64 for now - proper decimal support can come later
    // TODO: Implement me!
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, static_cast<int64_t>(value));
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_adddecimal OUT");
}

// Fixed-size binary type
void TableBuilder::addFixedSized(bool is_valid, int64_t value) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addfixedsized IN: value=%ld, is_valid=%s", value, is_valid ? "true" : "false");
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, value);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_addfixedsized OUT");
}

void TableBuilder::destroy(void* builder) {
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_destroy IN: builder=%p", builder);
    // Note: We don't need to do anything here because the MemoryContextCallback
    // will handle cleanup when the memory context is reset/deleted.
    // Calling destructor or pfree here would cause double-free issues.
    PGX_LOG(RUNTIME, DEBUG, "rt_tablebuilder_destroy called - cleanup handled by MemoryContextCallback");
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_destroy OUT");
}

// Memory context callback for DataSourceIterator cleanup
static void cleanup_datasourceiterator_callback(void* arg) {
    DataSourceIterator* iter = static_cast<DataSourceIterator*>(arg);
    if (iter) {
        iter->~DataSourceIterator();  // Explicit destructor call for PostgreSQL longjmp safety
    }
}

// TODO: This function is uhhh... pretty gross. It should be returning iter, not a boolean. I also cannot be bothered fixing
// it now since it does its job and its just an abstracted away black box
static bool decode_table_specification(runtime::VarLen32 varlen32_param, DataSourceIterator* iter) {
    uint32_t actual_len = varlen32_param.getLen();
    const char* json_spec = varlen32_param.data();

    PGX_LOG(RUNTIME, DEBUG, "decode_table_specification: LingoDB runtime::VarLen32 len=%u", actual_len);

    if (!json_spec || actual_len == 0) {
        return false;
    }

    bool json_parsed = false;
    PG_TRY();
    {
        std::string json_string(json_spec, actual_len);
        PGX_LOG(RUNTIME, DEBUG, "decode_table_specification: JSON string: %s", json_string.c_str());

        if (json_string[0] == '{') {
            PGX_LOG(RUNTIME, DEBUG, "decode_table_specification: valid JSON detected, parsing...");
            TableSpec spec = parse_table_spec(json_string.c_str());

            if (!spec.table_name.empty()) {
                size_t pipe_pos = spec.table_name.find("|oid:");
                if (pipe_pos != std::string::npos) {
                    std::string oid_str = spec.table_name.substr(pipe_pos + 5); // Skip "|oid:"
                    Oid table_oid = static_cast<Oid>(std::stoul(oid_str));
                    ::g_jit_table_oid = table_oid;
                    PGX_LOG(RUNTIME, DEBUG, "Extracted table OID %u from spec", table_oid);
                    
                    spec.table_name = spec.table_name.substr(0, pipe_pos);
                } else {
                    PGX_LOG(RUNTIME, DEBUG, "No OID in table spec, g_jit_table_oid unchanged");
                }

                iter->table_name = spec.table_name;

                // Get all column metadata in one shot to avoid repeated table opens
                extern int32_t get_all_column_metadata(const char* table_name, ColumnMetadata* metadata, int32_t max_columns);

                // Use PostgreSQL's maximum column limit
                ColumnMetadata metadata[MaxTupleAttributeNumber];
                int32_t total_columns = get_all_column_metadata(spec.table_name.c_str(), metadata, MaxTupleAttributeNumber);

                if (total_columns <= 0) {
                    PGX_ERROR("Failed to get column metadata for table '%s'", spec.table_name.c_str());
                    throw std::runtime_error("Failed to get table metadata");
                }

                PGX_LOG(RUNTIME, DEBUG, "Retrieved metadata for %d columns from table '%s'",
                        total_columns, spec.table_name.c_str());

                if (spec.column_names.empty()) {
                    PGX_LOG(RUNTIME, DEBUG, "No specific columns requested (e.g., COUNT(*)), skipping column metadata processing");
                    json_parsed = true;
                    iter->table_name = spec.table_name;
                } else {
                    for (size_t i = 0; i < spec.column_names.size(); ++i) {
                    ColumnSpec col_spec;
                    col_spec.name = spec.column_names[i];

                    int32_t type_oid = 0;
                    for (int32_t j = 0; j < total_columns; ++j) {
                        if (strcmp(metadata[j].name, col_spec.name.c_str()) == 0) {
                            type_oid = metadata[j].type_oid;
                            break;
                        }
                    }

                    if (type_oid == 0) {
                        PGX_ERROR("Column '%s' not found in table '%s' metadata",
                                  col_spec.name.c_str(), spec.table_name.c_str());
                        throw std::runtime_error("Column not found in table");
                    }

                    switch (type_oid) {
                    case BOOLOID:
                        col_spec.type = ::ColumnType::BOOLEAN;
                        break;
                    case INT8OID:
                        col_spec.type = ::ColumnType::BIGINT;
                        break;
                    case TEXTOID:
                    case VARCHAROID:
                    case BPCHAROID:
                        col_spec.type = ::ColumnType::STRING;
                        break;
                    case INT4OID:
                    case INT2OID:
                        col_spec.type = ::ColumnType::INTEGER;
                        break;
                    default:
                        PGX_ERROR("Unsupported type %d for column '%s'", type_oid, col_spec.name.c_str());
                        throw std::runtime_error("Failed to parse column type");
                    }

                    iter->columns.push_back(col_spec);
                    }

                    json_parsed = true;
                    PGX_LOG(RUNTIME, DEBUG, "decode_table_specification: JSON parsed successfully - table '%s' with %zu columns", iter->table_name.c_str(), iter->columns.size());
                }
            }
        }
    }
    PG_CATCH();
    {
        PGX_ERROR( "decode_table_specification: exception reading runtime::VarLen32 JSON");
        FlushErrorState();
        throw std::runtime_error("Failed to decode table specification");
    }
    PG_END_TRY();

    return json_parsed;
}

// Helper function: Initialize column storage for dynamic types
static void initialize_column_storage(DataSourceIterator* iter) {
    // Reserve capacity to prevent reallocation during iteration
    iter->int_values.reserve(10000);
    iter->int_nulls.reserve(10000);
    iter->bigint_values.reserve(10000);
    iter->bigint_nulls.reserve(10000);
    iter->bool_values.reserve(10000);
    iter->bool_nulls.reserve(10000);

    iter->int_values.resize(iter->columns.size(), 0);
    iter->int_nulls.resize(iter->columns.size(), true);
    iter->bigint_values.resize(iter->columns.size(), 0);
    iter->bigint_nulls.resize(iter->columns.size(), true);
    iter->bool_values.resize(iter->columns.size(), 0);
    iter->bool_nulls.resize(iter->columns.size(), true);

    iter->string_data.resize(iter->columns.size(), "");
    iter->string_nulls.resize(iter->columns.size(), true);
    
    // Initialize per-column string storage
    iter->string_offsets_per_column.resize(iter->columns.size());
    iter->string_data_buffers_per_column.resize(iter->columns.size());
    
    // Reserve capacity for each string column
    for (size_t i = 0; i < iter->columns.size(); ++i) {
        if (iter->columns[i].type == ::ColumnType::STRING || 
            iter->columns[i].type == ::ColumnType::TEXT || 
            iter->columns[i].type == ::ColumnType::VARCHAR) {
            iter->string_offsets_per_column[i].reserve(10000);  // Reserve for up to 10k rows
            iter->string_data_buffers_per_column[i].reserve(100000);  // Reserve 100KB per column
        }
    }

    PGX_LOG(RUNTIME, DEBUG, "initialize_column_storage: configured for table '%s' with %zu columns", iter->table_name.c_str(), iter->columns.size());
}

// Helper function: Open PostgreSQL table connection
static void* open_table_connection(const std::string& table_name) {
    void* table_handle = open_postgres_table(table_name.c_str());

    if (!table_handle) {
        PGX_LOG(RUNTIME, DEBUG, "open_table_connection: open_postgres_table failed for '%s'", table_name.c_str());
    }

    return table_handle;
}

DataSourceIteration* DataSourceIteration::start(ExecutionContext* context, runtime::VarLen32 varlen32_param) {
    PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_start IN: context=%p, varlen32_param len=%u", context, varlen32_param.getLen());
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_start called with context: %p, varlen32_param len: %u", context, varlen32_param.getLen());

    // Use PostgreSQL memory management instead of malloc()
    MemoryContext oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    void* iter_memory = palloc(sizeof(DataSourceIterator));
    DataSourceIterator* iter = new(iter_memory) DataSourceIterator();

    // Register cleanup callback for C++ destructor safety during PostgreSQL error recovery
    MemoryContextCallback* callback = (MemoryContextCallback*)palloc(sizeof(MemoryContextCallback));
    callback->func = cleanup_datasourceiterator_callback;
    callback->arg = iter;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

    MemoryContextSwitchTo(oldcontext);

    // Initialize basic iterator state
    iter->context = context;
    iter->has_current_tuple = false;
    
    PGX_LOG(RUNTIME, DEBUG, "Before clear: string_offsets_per_column.size()=%zu", iter->string_offsets_per_column.size());
    iter->current_row_index = 0;
    iter->string_offsets_per_column.clear();
    iter->string_data_buffers_per_column.clear();
    iter->string_data.clear();
    iter->string_nulls.clear();
    PGX_LOG(RUNTIME, DEBUG, "After clear: string_offsets_per_column.size()=%zu", iter->string_offsets_per_column.size());
    
    iter->current_value = 0;
    iter->current_is_null = true;

    // Decode table specification from runtime::VarLen32 parameter
    bool json_parsed = decode_table_specification(varlen32_param, iter);

    if (!json_parsed) {
        PGX_ERROR("JSON parsing failed");
        throw std::runtime_error("Failed to parse the json");
    }

    // Initialize column storage based on parsed configuration
    initialize_column_storage(iter);

    // Open PostgreSQL table connection
    iter->table_handle = open_table_connection(iter->table_name);

    if (!iter->table_handle) {
        return reinterpret_cast<DataSourceIteration*>(iter);
    }

    g_current_iterator = iter;
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_start returning iterator: %p", iter);
    PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_start OUT: iterator=%p", iter);
    return reinterpret_cast<DataSourceIteration*>(iter);
}

bool DataSourceIteration::isValid() {
    PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_isvalid IN: iterator=%p", this);
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid called with iterator: %p", this);
    if (!this) return false;

    auto* iter = (DataSourceIterator*)this;
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: iter=%p, table_handle=%p", iter, iter->table_handle);
    if (!iter->table_handle) {
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid finished running with: %p branch 1 (no table_handle)", this);
        PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_isvalid OUT: false (no table_handle)");
        return false;
    }
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: About to call read_next_tuple_from_table");
    int64_t read_result = read_next_tuple_from_table(iter->table_handle);
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: read_result = %ld", read_result);

    if (read_result == 1) {
        extern int32_t get_int32_field(void* tuple_handle, int32_t field_index, bool* is_null);
        extern int64_t get_int64_field(void* tuple_handle, int32_t field_index, bool* is_null);
        extern bool get_bool_field(void* tuple_handle, int32_t field_index, bool* is_null);
        extern const char* get_string_field(void* tuple_handle, int32_t field_index, bool* is_null, int32_t* length, int32_t type_oid);
        extern int32_t get_field_type_oid(int32_t field_index);

        // Read dynamic columns based on specification
        // We need to map column names to their actual PostgreSQL positions
        // because MLIR may have columns in different order than PostgreSQL physical order
        for (size_t i = 0; i < iter->columns.size(); ++i) {
            auto& col_spec = iter->columns[i];  // Non-const so we can update the type

            // Find the actual PostgreSQL column index for this column name
            int pg_column_index = get_column_position(iter->table_name, col_spec.name);

            if (pg_column_index == -1) {
                PGX_ERROR("rt_datasourceiteration_isvalid: No mapping found for column '%s' in table '%s', crashing",
                     col_spec.name.c_str(), iter->table_name.c_str());
                pg_column_index = i;
            }

            // Get the actual type OID from PostgreSQL
            int32_t type_oid = get_field_type_oid(pg_column_index);

            // Determine the column type from the PostgreSQL type OID
            if (type_oid == 16) {  // BOOLOID
                col_spec.type = ::ColumnType::BOOLEAN;
                bool is_null = false;
                bool bool_value = get_bool_field(iter->table_handle, pg_column_index, &is_null);
                iter->bool_values[i] = bool_value ? 1 : 0;
                iter->bool_nulls[i] = is_null;
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: column %zu (%s) from PG column %d = %s (null=%s)",
                     i, col_spec.name.c_str(), pg_column_index, bool_value ? "true" : "false", is_null ? "true" : "false");
            } else if (col_spec.type == ::ColumnType::BIGINT) {
                bool is_null = false;
                iter->bigint_values[i] = get_int64_field(iter->table_handle, pg_column_index, &is_null);
                iter->bigint_nulls[i] = is_null;
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: column %zu (%s) from PG column %d = %lld (null=%s)",
                     i, col_spec.name.c_str(), pg_column_index, (long long)iter->bigint_values[i], is_null ? "true" : "false");
            } else if (col_spec.type == ::ColumnType::STRING || col_spec.type == ::ColumnType::TEXT || col_spec.type == ::ColumnType::VARCHAR) {
                bool is_null = false;
                int32_t length = 0;
                const char* string_value = get_string_field(iter->table_handle, pg_column_index, &is_null, &length, type_oid);

                if (!is_null && string_value != nullptr && length > 0) {
                    iter->string_data[i] = std::string(string_value, length);
                } else {
                    iter->string_data[i] = "";
                }

                iter->string_nulls[i] = is_null;

                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: column %zu (%s) from PG column %d string = '%s' len=%zu (null=%s)",
                     i, col_spec.name.c_str(), pg_column_index,
                     iter->string_data[i].c_str(), iter->string_data[i].length(),
                     is_null ? "true" : "false");
            } else {
                bool is_null = false;
                iter->int_values[i] = get_int32_field(iter->table_handle, pg_column_index, &is_null);
                iter->int_nulls[i] = is_null;
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: column %zu (%s) from PG column %d = %d (null=%s)",
                     i, col_spec.name.c_str(), pg_column_index, iter->int_values[i], is_null ? "true" : "false");
            }
        }
        
        // Build per-column offset arrays and data buffers for string columns
        if (iter->current_row_index == 0) {
            for (size_t i = 0; i < iter->columns.size(); ++i) {
                if (iter->columns[i].type == ::ColumnType::STRING || 
                    iter->columns[i].type == ::ColumnType::TEXT || 
                    iter->columns[i].type == ::ColumnType::VARCHAR) {
                    if (iter->string_offsets_per_column[i].empty()) {
                        iter->string_offsets_per_column[i].push_back(0);
                        PGX_LOG(RUNTIME, DEBUG, "Added initial offset 0 for column %zu (%s)", 
                                i, iter->columns[i].name.c_str());
                    }
                }
            }
        }
        
        for (size_t i = 0; i < iter->columns.size(); ++i) {
            if ((iter->columns[i].type == ::ColumnType::STRING || 
                 iter->columns[i].type == ::ColumnType::TEXT || 
                 iter->columns[i].type == ::ColumnType::VARCHAR)) {
                
                if (!iter->string_data[i].empty()) {
                    iter->string_data_buffers_per_column[i].insert(
                        iter->string_data_buffers_per_column[i].end(),
                        iter->string_data[i].begin(),
                        iter->string_data[i].end()
                    );
                }
                
                iter->string_offsets_per_column[i].push_back(iter->string_data_buffers_per_column[i].size());
                
                size_t offset_count = iter->string_offsets_per_column[i].size();
                PGX_LOG(RUNTIME, DEBUG, "Column %zu (%s) after row %d: offset_count=%zu, data_buffer.size()=%zu",
                        i, iter->columns[i].name.c_str(), iter->current_row_index, 
                        offset_count, iter->string_data_buffers_per_column[i].size());
                
                if (offset_count >= 2) {
                    size_t curr_idx = iter->current_row_index;
                    if (curr_idx < offset_count - 1) {
                        PGX_LOG(RUNTIME, DEBUG, "Column %zu offsets for row %d: offsets[%zu]=%d, offsets[%zu]=%d",
                                i, iter->current_row_index,
                                curr_idx, iter->string_offsets_per_column[i][curr_idx],
                                curr_idx + 1, iter->string_offsets_per_column[i][curr_idx + 1]);
                    }
                }
            }
        }
        
        iter->current_row_index++;

        // Maintain legacy fields for backward compatibility
        if (iter->columns.size() >= 1 && iter->columns[0].type == ::ColumnType::INTEGER) {
            iter->current_id = iter->int_values[0];
            iter->current_id_is_null = iter->int_nulls[0];
            iter->current_value = iter->current_id;
            iter->current_is_null = iter->current_id_is_null;
        }
        if (iter->columns.size() >= 2 && iter->columns[1].type == ::ColumnType::INTEGER) {
            iter->current_col2 = iter->int_values[1];
            iter->current_col2_is_null = iter->int_nulls[1];
        }

        iter->has_current_tuple = true;
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid finished running with: %p branch 2", this);
        PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_isvalid OUT: true");
        return true;
    } else {
        iter->has_current_tuple = false;
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid finished running with: %p branch 3", this);
        PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_isvalid OUT: false");
        return false;
    }
}

void DataSourceIteration::access(RecordBatchInfo* info) {
    auto *row_data = info;
    PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_access IN: iterator=%p, row_data=%p", this, row_data);
    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access called with iterator: %p, row_data: %p", this, row_data);
    if (row_data) {
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: row_data is valid, proceeding");

        auto* iter = (DataSourceIterator*)this;
        if (!iter || !iter->has_current_tuple) {
            PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: invalid iterator or no current tuple");
            return;
        }

        size_t num_columns = iter->columns.size();
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: handling %zu columns", num_columns);

        // The row_data is expected to be a dynamic structure:
        // [numRows: size_t][columnInfo[0]: COLUMN_INFO_SIZE*size_t][columnInfo[1]: COLUMN_INFO_SIZE*size_t]...[columnInfo[n]: COLUMN_INFO_SIZE*size_t]
        size_t* row_data_ptr = (size_t*)row_data;
        row_data_ptr[0] = 1; // numRows = 1

        // Create validity bitmaps for each column based on actual null flags
        static std::vector<uint8_t> valid_bitmaps;
        valid_bitmaps.resize(num_columns);

        for (size_t i = 0; i < num_columns; ++i) {
            constexpr size_t COLUMN_OFFSET_IDX = 0;
            constexpr size_t VALID_MULTIPLIER_IDX = 1;
            constexpr size_t VALID_BUFFER_IDX = 2;
            constexpr size_t DATA_BUFFER_IDX = 3;
            constexpr size_t VARLEN_BUFFER_IDX = 4;
            constexpr size_t COLUMN_INFO_SIZE = 5;

            const auto& col_spec = iter->columns[i];
            size_t* column_info_ptr = &row_data_ptr[1 + i * COLUMN_INFO_SIZE]; // Each ColumnInfo has COLUMN_INFO_SIZE fields

            // Set validity bitmap based on whether the value is NULL
            // Note: In LingoDB, 1 means valid (not null), 0 means null
            bool is_null;
            if (col_spec.type == ::ColumnType::BOOLEAN) {
                is_null = iter->bool_nulls[i];
            } else if (col_spec.type == ::ColumnType::BIGINT) {
                is_null = iter->bigint_nulls[i];
            } else if (col_spec.type == ::ColumnType::STRING || col_spec.type == ::ColumnType::TEXT || col_spec.type == ::ColumnType::VARCHAR) {
                is_null = iter->string_nulls[i];
            } else {
                is_null = iter->int_nulls[i];
            }
            valid_bitmaps[i] = is_null ? 0x00 : 0xFF;  // 0x00 for null, 0xFF for valid

            PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu is_null=%s, valid_bitmap=0x%02X",
                 i, is_null ? "true" : "false", valid_bitmaps[i]);

            column_info_ptr[COLUMN_OFFSET_IDX] = 0;                    // offset
            column_info_ptr[VALID_MULTIPLIER_IDX] = 0;                 // validMultiplier
            column_info_ptr[VALID_BUFFER_IDX] = (size_t)&valid_bitmaps[i]; // validBuffer - unique per column

            if (col_spec.type == ::ColumnType::BOOLEAN) {
                column_info_ptr[VARLEN_BUFFER_IDX] = 0;                    // varLenBuffer = nullptr for non-string types
                column_info_ptr[DATA_BUFFER_IDX] = (size_t)&iter->bool_values[i]; // dataBuffer
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu (%s) boolean = %s at %p",
                     i, col_spec.name.c_str(), iter->bool_values[i] ? "true" : "false", &iter->bool_values[i]);
            } else if (col_spec.type == ::ColumnType::BIGINT) {
                column_info_ptr[VARLEN_BUFFER_IDX] = 0;                    // varLenBuffer = nullptr for non-string types
                column_info_ptr[DATA_BUFFER_IDX] = (size_t)&iter->bigint_values[i]; // dataBuffer
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu (%s) bigint = %lld at %p",
                     i, col_spec.name.c_str(), (long long)iter->bigint_values[i], &iter->bigint_values[i]);
            } else if (col_spec.type == ::ColumnType::STRING || col_spec.type == ::ColumnType::TEXT || col_spec.type == ::ColumnType::VARCHAR) {
                // For string columns, provide a pointer to the CURRENT ROW's offset in the array
                // The row index tells us which pair of offsets to use
                size_t row_offset_index = iter->current_row_index - 1;  // -1 because we already incremented
                
                // Point to the offset for the current row (not the base of the array)
                // This gives LingoDB access to [start_offset, end_offset] for this row's string
                int32_t* row_offset_ptr = &iter->string_offsets_per_column[i][row_offset_index];
                column_info_ptr[DATA_BUFFER_IDX] = (size_t)row_offset_ptr;
                column_info_ptr[VARLEN_BUFFER_IDX] = (size_t)iter->string_data_buffers_per_column[i].data();

                if (row_offset_index < iter->string_offsets_per_column[i].size() - 1) {
                    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu (%s) providing row %zu offset ptr at %p (offsets[%zu]=%d, offsets[%zu]=%d), data at %p",
                         i, col_spec.name.c_str(), row_offset_index,
                         row_offset_ptr,
                         row_offset_index, iter->string_offsets_per_column[i][row_offset_index],
                         row_offset_index + 1, iter->string_offsets_per_column[i][row_offset_index + 1],
                         iter->string_data_buffers_per_column[i].data());
                } else {
                    PGX_LOG(RUNTIME, DEBUG, "WARNING: Invalid row offset index %zu for string column %s", row_offset_index, col_spec.name.c_str());
                }
            } else {
                column_info_ptr[VARLEN_BUFFER_IDX] = 0;                    // varLenBuffer = nullptr for non-string types
                column_info_ptr[DATA_BUFFER_IDX] = (size_t)&iter->int_values[i]; // dataBuffer
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu (%s) integer = %d at %p",
                     i, col_spec.name.c_str(), iter->int_values[i], &iter->int_values[i]);
            }
        }

        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: configured %zu columns successfully", num_columns);
    } else {
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: row_data is NULL");
    }

    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access completed successfully");
    PGX_LOG(RUNTIME, IO, "rt_datasourceiteration_access OUT");
    __sync_synchronize();
}

void DataSourceIteration::next() {
    auto* iter = (DataSourceIterator*)this;
    iter->has_current_tuple = false;
}

void DataSourceIteration::end(DataSourceIteration* iterator) {
    if (iterator) {
        auto* iter = (DataSourceIterator*)iterator;

        if (iter->table_handle) {
            extern void close_postgres_table(void* tableHandle);
            close_postgres_table(iter->table_handle);
            iter->table_handle = nullptr;
        }
        if (g_current_iterator == (DataSourceIterator*)iterator) {
            g_current_iterator = nullptr;
        }

        // Note: We don't call destructor or pfree here because the MemoryContextCallback
        // will handle cleanup when the memory context is reset/deleted.
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_end: cleanup will be handled by MemoryContextCallback");
    }
}

} // extern "C"

// Global context functions
void* getExecutionContext() {
    return ::rt_get_execution_context();  // Call the global C function
}

void setExecutionContext(void* context) {
    ::rt_set_execution_context(context);  // Call the global C function
}

} // namespace runtime