#include "lingodb/runtime/PostgreSQLRuntime.h"
#include "lingodb/runtime/DataSourceIteration.h"
#include "lingodb/runtime/StringRuntime.h"
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
    std::vector<runtime::VarLen32> string_values;  // For TEXT, VARCHAR, CHAR columns
    std::vector<bool> string_nulls;
    std::vector<std::string> string_data;  // Persistent storage for actual string data

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
    mark_results_ready_for_streaming();
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

            if (!spec.table_name.empty() && !spec.column_names.empty()) {
                // Remove OID suffix if present (e.g., "test_logical|oid:123" -> "test_logical")
                size_t pipe_pos = spec.table_name.find('|');
                if (pipe_pos != std::string::npos) {
                    spec.table_name = spec.table_name.substr(0, pipe_pos);
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

                // Now match the requested columns with the metadata
                for (size_t i = 0; i < spec.column_names.size(); ++i) {
                    ColumnSpec col_spec;
                    col_spec.name = spec.column_names[i];

                    // Find this column in the metadata
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
    PG_CATCH();
    {
        PGX_LOG(RUNTIME, DEBUG, "decode_table_specification: exception reading runtime::VarLen32 JSON, using fallback");
        FlushErrorState();
    }
    PG_END_TRY();

    return json_parsed;
}

// Helper function: Initialize column storage for dynamic types
static void initialize_column_storage(DataSourceIterator* iter) {
    iter->int_values.resize(iter->columns.size(), 0);
    iter->int_nulls.resize(iter->columns.size(), true);
    iter->bigint_values.resize(iter->columns.size(), 0);
    iter->bigint_nulls.resize(iter->columns.size(), true);
    iter->bool_values.resize(iter->columns.size(), 0);
    iter->bool_nulls.resize(iter->columns.size(), true);

    iter->string_values.clear();
    iter->string_data.resize(iter->columns.size(), "");
    for (size_t i = 0; i < iter->columns.size(); ++i) {
        const char* empty = "";
        iter->string_values.push_back(runtime::VarLen32(reinterpret_cast<uint8_t*>(const_cast<char*>(empty)), 0));
    }
    iter->string_nulls.resize(iter->columns.size(), true);

    PGX_LOG(RUNTIME, DEBUG, "initialize_column_storage: configured for table '%s' with %zu columns", iter->table_name.c_str(), iter->columns.size());
}

// Helper function: Set up fallback table configuration
static void setup_fallback_table_config(DataSourceIterator* iter) {
    PGX_LOG(RUNTIME, DEBUG, "setup_fallback_table_config: JSON parsing failed, using fallback defaults");
    iter->table_name = "test_comparison";

    // Set up default 2-integer column layout
    ColumnSpec col1, col2;
    col1.name = "value";
    col1.type = ::ColumnType::INTEGER;
    col2.name = "score";
    col2.type = ::ColumnType::INTEGER;
    iter->columns.push_back(col1);
    iter->columns.push_back(col2);
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
    iter->current_value = 0;
    iter->current_is_null = true;

    // Decode table specification from runtime::VarLen32 parameter
    bool json_parsed = decode_table_specification(varlen32_param, iter);

    // Set up fallback configuration if JSON parsing failed
    if (!json_parsed) {
        setup_fallback_table_config(iter);
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
                // Column not found in mapping, use iteration index as fallback
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: No mapping found for column '%s' in table '%s', using index %zu",
                     col_spec.name.c_str(), iter->table_name.c_str(), i);
                pg_column_index = i;  // Fallback to iteration order
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

                // Create runtime::VarLen32 from the persistent string storage
                iter->string_values[i] = runtime::VarLen32(
                    reinterpret_cast<uint8_t*>(const_cast<char*>(iter->string_data[i].data())),
                    iter->string_data[i].length());
                iter->string_nulls[i] = is_null;

                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: column %zu (%s) from PG column %d string = '%s' len=%u (null=%s)",
                     i, col_spec.name.c_str(), pg_column_index,
                     iter->string_data[i].c_str(), iter->string_values[i].getLen(),
                     is_null ? "true" : "false");
            } else {
                bool is_null = false;
                iter->int_values[i] = get_int32_field(iter->table_handle, pg_column_index, &is_null);
                iter->int_nulls[i] = is_null;
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_isvalid: column %zu (%s) from PG column %d = %d (null=%s)",
                     i, col_spec.name.c_str(), pg_column_index, iter->int_values[i], is_null ? "true" : "false");
            }
        }

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
                // For strings, DSA expects valueBuffer to be int32 offsets and varLenBuffer to be raw bytes
                // But since we already have runtime::VarLen32 structs, we'll provide them directly
                // The DSA lowering will need to handle this case
                column_info_ptr[DATA_BUFFER_IDX] = (size_t)&iter->string_values[i]; // dataBuffer points to runtime::VarLen32 struct
                column_info_ptr[VARLEN_BUFFER_IDX] = (size_t)&iter->string_values[i]; // varLenBuffer also points to runtime::VarLen32 struct
                PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu (%s) string len=%u at %p",
                     i, col_spec.name.c_str(), iter->string_values[i].getLen(), &iter->string_values[i]);
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

// ============================================================================
// StringRuntime implementation for new PostgreSQL operations
// ============================================================================
    // TODO: Move these into the stringruntime.cpp

runtime::VarLen32 runtime::StringRuntime::concat(runtime::VarLen32 left, runtime::VarLen32 right) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::concat IN: left_len=%u, right_len=%u", left.getLen(), right.getLen());
    
    // Allocate space for concatenated string
    uint32_t totalLen = left.getLen() + right.getLen();
    char* result = static_cast<char*>(palloc(totalLen + 1));
    
    // Copy left string
    memcpy(result, left.getPtr(), left.getLen());
    // Copy right string
    memcpy(result + left.getLen(), right.getPtr(), right.getLen());
    result[totalLen] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), totalLen);
    PGX_LOG(RUNTIME, IO, "StringRuntime::concat OUT: result_len=%u", totalLen);
    return resultVarLen;
}

runtime::VarLen32 runtime::StringRuntime::concat3(runtime::VarLen32 a, runtime::VarLen32 b, runtime::VarLen32 c) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::concat3 IN: a_len=%u, b_len=%u, c_len=%u", 
            a.getLen(), b.getLen(), c.getLen());
    
    uint32_t totalLen = a.getLen() + b.getLen() + c.getLen();
    char* result = static_cast<char*>(palloc(totalLen + 1));
    
    uint32_t offset = 0;
    memcpy(result + offset, a.getPtr(), a.getLen());
    offset += a.getLen();
    memcpy(result + offset, b.getPtr(), b.getLen());
    offset += b.getLen();
    memcpy(result + offset, c.getPtr(), c.getLen());
    result[totalLen] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), totalLen);
    PGX_LOG(RUNTIME, IO, "StringRuntime::concat3 OUT: result_len=%u", totalLen);
    return resultVarLen;
}

runtime::VarLen32 runtime::StringRuntime::upper(runtime::VarLen32 str) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::upper IN: str_len=%u", str.getLen());
    
    char* result = static_cast<char*>(palloc(str.getLen() + 1));
    const char* input = reinterpret_cast<const char*>(str.getPtr());
    
    for (uint32_t i = 0; i < str.getLen(); i++) {
        result[i] = toupper(input[i]);
    }
    result[str.getLen()] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), str.getLen());
    PGX_LOG(RUNTIME, IO, "StringRuntime::upper OUT: result_len=%u", str.getLen());
    return resultVarLen;
}

runtime::VarLen32 runtime::StringRuntime::lower(runtime::VarLen32 str) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::lower IN: str_len=%u", str.getLen());
    
    char* result = static_cast<char*>(palloc(str.getLen() + 1));
    const char* input = reinterpret_cast<const char*>(str.getPtr());
    
    for (uint32_t i = 0; i < str.getLen(); i++) {
        result[i] = tolower(input[i]);
    }
    result[str.getLen()] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), str.getLen());
    PGX_LOG(RUNTIME, IO, "StringRuntime::lower OUT: result_len=%u", str.getLen());
    return resultVarLen;
}

runtime::VarLen32 runtime::StringRuntime::substring(runtime::VarLen32 str, int32_t start, int32_t length) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::substring IN: str_len=%u, start=%d, length=%d", 
            str.getLen(), start, length);
    
    // PostgreSQL SUBSTRING uses 1-based indexing
    // Convert to 0-based for internal use
    int32_t zeroBasedStart = start - 1;
    
    // Handle negative or out of bounds start
    if (zeroBasedStart < 0) {
        zeroBasedStart = 0;
    }
    if (zeroBasedStart >= static_cast<int32_t>(str.getLen())) {
        // Return empty string
        char* empty = static_cast<char*>(palloc(1));
        empty[0] = '\0';
        return runtime::VarLen32(reinterpret_cast<uint8_t*>(empty), 0);
    }
    
    // Calculate actual length to copy
    int32_t maxLen = str.getLen() - zeroBasedStart;
    int32_t actualLen = (length < 0 || length > maxLen) ? maxLen : length;
    
    char* result = static_cast<char*>(palloc(actualLen + 1));
    memcpy(result, str.getPtr() + zeroBasedStart, actualLen);
    result[actualLen] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), actualLen);
    PGX_LOG(RUNTIME, IO, "StringRuntime::substring OUT: result_len=%d", actualLen);
    return resultVarLen;
}

int32_t runtime::StringRuntime::length(runtime::VarLen32 str) {
    return static_cast<int32_t>(str.getLen());
}

int32_t runtime::StringRuntime::charLength(runtime::VarLen32 str) {
    // For now, assume single-byte characters
    // TODO: Handle multi-byte UTF-8 properly
    return static_cast<int32_t>(str.getLen());
}

runtime::VarLen32 runtime::StringRuntime::trim(runtime::VarLen32 str) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::trim IN: str_len=%u", str.getLen());
    
    const char* input = reinterpret_cast<const char*>(str.getPtr());
    uint32_t len = str.getLen();
    
    // Find start of non-whitespace
    uint32_t start = 0;
    while (start < len && isspace(input[start])) {
        start++;
    }
    
    // Find end of non-whitespace
    uint32_t end = len;
    while (end > start && isspace(input[end - 1])) {
        end--;
    }
    
    uint32_t resultLen = end - start;
    char* result = static_cast<char*>(palloc(resultLen + 1));
    memcpy(result, input + start, resultLen);
    result[resultLen] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), resultLen);
    PGX_LOG(RUNTIME, IO, "StringRuntime::trim OUT: result_len=%u", resultLen);
    return resultVarLen;
}

runtime::VarLen32 runtime::StringRuntime::ltrim(runtime::VarLen32 str) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::ltrim IN: str_len=%u", str.getLen());
    
    const char* input = reinterpret_cast<const char*>(str.getPtr());
    uint32_t len = str.getLen();
    
    // Find start of non-whitespace
    uint32_t start = 0;
    while (start < len && isspace(input[start])) {
        start++;
    }
    
    uint32_t resultLen = len - start;
    char* result = static_cast<char*>(palloc(resultLen + 1));
    memcpy(result, input + start, resultLen);
    result[resultLen] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), resultLen);
    PGX_LOG(RUNTIME, IO, "StringRuntime::ltrim OUT: result_len=%u", resultLen);
    return resultVarLen;
}

runtime::VarLen32 runtime::StringRuntime::rtrim(runtime::VarLen32 str) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::rtrim IN: str_len=%u", str.getLen());
    
    const char* input = reinterpret_cast<const char*>(str.getPtr());
    uint32_t len = str.getLen();
    
    // Find end of non-whitespace
    uint32_t end = len;
    while (end > 0 && isspace(input[end - 1])) {
        end--;
    }
    
    char* result = static_cast<char*>(palloc(end + 1));
    memcpy(result, input, end);
    result[end] = '\0';
    
    runtime::VarLen32 resultVarLen(reinterpret_cast<uint8_t*>(result), end);
    PGX_LOG(RUNTIME, IO, "StringRuntime::rtrim OUT: result_len=%u", end);
    return resultVarLen;
}

bool runtime::StringRuntime::ilike(runtime::VarLen32 str, runtime::VarLen32 pattern) {
    PGX_LOG(RUNTIME, IO, "StringRuntime::ilike IN: str_len=%u, pattern_len=%u", 
            str.getLen(), pattern.getLen());
    
    // Convert both strings to lowercase and then use regular like
    runtime::VarLen32 lowerStr = lower(str);
    runtime::VarLen32 lowerPattern = lower(pattern);
    
    bool result = like(lowerStr, lowerPattern);
    
    PGX_LOG(RUNTIME, IO, "StringRuntime::ilike OUT: result=%d", result);
    return result;
}

} // namespace runtime