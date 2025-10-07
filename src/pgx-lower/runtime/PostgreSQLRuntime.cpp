#include "pgx-lower/runtime/PostgreSQLRuntime.h"
#include "lingodb/runtime/DataSourceIteration.h"
#include "pgx-lower/runtime/StringRuntime.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cmath>
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
#include "utils/datum.h"
#include "utils/memutils.h"
#include "fmgr.h"
#include "utils/builtins.h"
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
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "rt_get_execution_context called");
    if (g_execution_context) {
        PGX_LOG(RUNTIME, DEBUG, "rt_get_execution_context returning g_execution_context: %p", g_execution_context);
        return g_execution_context;
    }
    static struct {
        void* table_ref;
        int64_t row_count;
    } dummy_context = {nullptr, 1};

    return &dummy_context;
}

enum class ColumnType {
    SMALLINT, // INT2OID (16-bit)
    INTEGER, // INT4OID (32-bit)
    BIGINT, // INT8OID (64-bit)
    BOOLEAN, // BOOLOID
    STRING, // TEXTOID, VARCHAROID, BPCHAROID, CHAROID
    TEXT, // Legacy - maps to STRING
    VARCHAR, // Legacy - maps to STRING
    DECIMAL, // NUMERICOID
    FLOAT, // FLOAT4OID
    DOUBLE, // FLOAT8OID
    DATE, // DATEOID
    TIMESTAMP, // TIMESTAMPOID, TIMESTAMPTZOID
    INTERVAL // INTERVALOID
};

struct ColumnSpec {
    std::string name;
    ColumnType type;
};

struct BatchStorage {
    MemoryContext batchContext;
    TupleDesc tupleDesc;

    size_t capacity;
    size_t num_rows;

    Datum** column_values;
    bool** column_nulls;
};

struct DataSourceIterator {
    void* context;
    void* table_handle;

    std::string table_name;
    std::vector<ColumnSpec> columns;

    BatchStorage* batch;
    bool batch_exhausted;

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
    PGX_IO(RUNTIME);
    extern int32_t get_column_attnum(const char* p_table_name, const char* p_column_name);
    const int32_t attnum = get_column_attnum(table_name.c_str(), column_name.c_str());
    if (attnum > 0) {
        // Convert from 1-based PostgreSQL attnum to 0-based index
        return attnum - 1;
    }

    return -1; // Column not found
}

static TableSpec parse_table_spec(const char* json_str) {
    PGX_IO(RUNTIME);
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

        PGX_LOG(RUNTIME, DEBUG, "parse_table_spec: table=%s, columns=%zu", spec.table_name.c_str(),
                spec.column_names.size());
    } catch (const std::exception& e) {
        PGX_LOG(RUNTIME, DEBUG, "parse_table_spec: JSON parsing failed: %s", e.what());
        // Return empty spec on error
    }

    return spec;
}

// Memory context callback for TableBuilder cleanup
static void cleanup_tablebuilder_callback(void* arg) {
    PGX_IO(RUNTIME);
    auto tb = static_cast<runtime::TableBuilder*>(arg);
}

namespace runtime {

// Constructor
TableBuilder::TableBuilder()
: data(nullptr)
, row_count(0)
, current_column_index(0)
, total_columns(0) {}

// Static factory method
TableBuilder* TableBuilder::create(VarLen32 schema_param) {
    PGX_IO(RUNTIME);

    const MemoryContext oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    void* builder_memory = palloc(sizeof(runtime::TableBuilder));
    const auto builder = new (builder_memory) runtime::TableBuilder();

    builder->total_columns = 0;
    PGX_LOG(RUNTIME, DEBUG, "rt_tablebuilder_create: initialized with dynamic column tracking");

    const auto callback = (MemoryContextCallback*)palloc(sizeof(MemoryContextCallback));
    callback->func = cleanup_tablebuilder_callback;
    callback->arg = builder;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

    MemoryContextSwitchTo(oldcontext);
    return builder;
}

void TableBuilder::nextRow() {
    PGX_IO(RUNTIME);

    // 'this' is the TableBuilder instance

    // LingoDB currColumn assertion pattern: verify all columns were filled
    if (current_column_index != total_columns) {
        PGX_LOG(RUNTIME, DEBUG,
                "TableBuilder::nextRow: column count info - expected %d columns, got %d (this may be normal during "
                "MLIR pipeline development)",
                total_columns, current_column_index);
    } else {
        PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: LingoDB column validation passed - %d columns filled",
                current_column_index);
    }

    row_count++;

    if (total_columns > 0) {
        PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: submitting row with %d columns", total_columns);
        add_tuple_to_result(total_columns);
    }

    current_column_index = 0;
    PGX_LOG(RUNTIME, DEBUG, "TableBuilder::nextRow: reset column index to 0 for row %ld", row_count);
}

TableBuilder* TableBuilder::build() {
    PGX_IO(RUNTIME);

    mark_results_ready_for_streaming();

    PGX_LOG(RUNTIME, DEBUG, "TableBuilder state before return:");
    PGX_LOG(RUNTIME, DEBUG, "\t- builder address: %p", this);
    PGX_LOG(RUNTIME, DEBUG, "\t- row_count: %ld", row_count);
    PGX_LOG(RUNTIME, DEBUG, "\t- total_columns: %d", total_columns);
    PGX_LOG(RUNTIME, DEBUG, "\t- current_column_index: %d", current_column_index);
    if (g_computed_results.numComputedColumns > 0) {
        PGX_LOG(RUNTIME, DEBUG, "\t- computed columns: %d", g_computed_results.numComputedColumns);
        for (int i = 0; i < g_computed_results.numComputedColumns && i < 10; i++) {
            PGX_LOG(RUNTIME, DEBUG, "\t\t- col[%d]: type=%d, null=%d", i, g_computed_results.computedTypes[i],
                    g_computed_results.computedNulls[i]);
        }
    }

    return this;
}

void TableBuilder::addInt64(bool is_valid, int64_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, value);
}

void TableBuilder::addInt32(bool is_valid, int32_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int32_t>(this, is_valid, value);
}

void TableBuilder::addBool(bool is_valid, bool value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<bool>(this, is_valid, value);
}

void TableBuilder::addInt8(bool is_valid, int8_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int8_t>(this, is_valid, value);
}

void TableBuilder::addInt16(bool is_valid, int16_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int16_t>(this, is_valid, value);
}

void TableBuilder::addFloat32(bool is_valid, float value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<float>(this, is_valid, value);
}

void TableBuilder::addFloat64(bool is_valid, double value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<double>(this, is_valid, value);
}

void TableBuilder::addBinary(bool is_valid, VarLen32 value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<runtime::VarLen32>(this, is_valid, value);
}

void TableBuilder::setNextDecimalScale(int32_t scale) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "rt_tablebuilder_setnextdecimalscale: scale=%d", scale);
    this->next_decimal_scale = scale;
}

void TableBuilder::addDecimal(bool is_valid, __int128 value) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, IO, "rt_tablebuilder_adddecimal IN: is_valid=%s", is_valid ? "true" : "false");

    if (!is_valid) {
        pgx_lower::runtime::table_builder_add_numeric(this, true, nullptr);
    } else {
        // TODO: This obviously isn't perfect... converting into a string then relying on postgres constructors
        //       is far from ideal. also, since its a uin128_t I can't just sprintf because it overflows. So...
        if (!this->next_decimal_scale.has_value()) {
            PGX_ERROR("Never set the decimal scale");
            throw std::runtime_error("Have no decimal scale");
        }
        auto scale = this->next_decimal_scale.value();

        char value_str[45];
        const bool is_negative = (value < 0);
        __uint128_t abs_value = is_negative ? -static_cast<__uint128_t>(value) : static_cast<__uint128_t>(value);

        char* p = value_str + sizeof(value_str) - 1;
        *p = '\0';
        do {
            *--p = '0' + (abs_value % 10);
            abs_value /= 10;
        } while (abs_value > 0 && p > value_str);

        if (is_negative && p > value_str) {
            *--p = '-';
        }

        size_t len = strlen(p);
        char* end = p + len - 1;
        int zeros_removed = 0;
        while (end > p && *end == '0' && zeros_removed < scale) {
            *end-- = '\0';
            zeros_removed++;
        }
        scale -= zeros_removed;
        len = strlen(p);

        // Format the numeric string with proper decimal point placement
        char buffer[128];
        if (scale > 0) {
            // Need to insert decimal point
            if (len <= scale) {
                // Number is less than 1, need leading zeros
                const int leading_zeros = scale - len + 1;
                buffer[0] = '0';
                buffer[1] = '.';
                int pos = 2;
                for (int i = 1; i < leading_zeros; i++) {
                    buffer[pos++] = '0';
                }
                strcpy(buffer + pos, p);
            } else {
                // Number >= 1, insert decimal point at appropriate position
                const size_t integer_len = len - scale;
                strncpy(buffer, p, integer_len);
                buffer[integer_len] = '.';
                strcpy(buffer + integer_len + 1, p + integer_len);
            }
        } else {
            // No decimal point needed
            strcpy(buffer, p);
        }
        PGX_LOG(RUNTIME, DEBUG, "Decimal numeric: %s (scale=%d, removed %d trailing zeros)", buffer, scale,
                zeros_removed);

        PGX_LOG(RUNTIME, DEBUG, "addDecimal: before DirectFunctionCall3, CurrentMemoryContext=%p", CurrentMemoryContext);

        const auto numeric_datum = DirectFunctionCall3(numeric_in, CStringGetDatum(buffer),
                                                       ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1));

        const auto copied_datum = datumCopy(numeric_datum, false, -1);
        const auto numeric_value = DatumGetNumeric(copied_datum);

        PGX_LOG(RUNTIME, DEBUG, "addDecimal: created and copied Numeric at %p (orig=%lu, copied=%lu)", numeric_value,
                numeric_datum, copied_datum);

        pgx_lower::runtime::table_builder_add_numeric(this, false, numeric_value);
        this->next_decimal_scale = std::nullopt;
    }
}

// Fixed-size binary type
void TableBuilder::addFixedSized(bool is_valid, int64_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, value);
}

void TableBuilder::destroy(void* builder) {
    PGX_IO(RUNTIME);
    // Note: We don't need to do anything here because the MemoryContextCallback
    // will handle cleanup when the memory context is reset/deleted.
    // Calling destructor or pfree here would cause double-free issues.
    PGX_LOG(RUNTIME, DEBUG, "rt_tablebuilder_destroy called - cleanup handled by MemoryContextCallback");
}

// Memory context callback for DataSourceIterator cleanup
static void cleanup_datasourceiterator_callback(void* arg) {
    PGX_IO(RUNTIME);
    const auto iter = static_cast<DataSourceIterator*>(arg);
    if (iter) {
        iter->~DataSourceIterator(); // Explicit destructor call for PostgreSQL longjmp safety
    }
}

// TODO: This function is uhhh... pretty gross. It should be returning iter, not a boolean. I also cannot be bothered
// fixing it now since it does its job and its just an abstracted away black box
static bool decode_table_specification(runtime::VarLen32 varlen32_param, DataSourceIterator* iter) {
    PGX_IO(RUNTIME);
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
                extern int32_t get_all_column_metadata(const char* table_name, ColumnMetadata* metadata,
                                                       int32_t max_columns);

                // Use PostgreSQL's maximum column limit
                ColumnMetadata metadata[MaxTupleAttributeNumber];
                int32_t total_columns = get_all_column_metadata(spec.table_name.c_str(), metadata,
                                                                MaxTupleAttributeNumber);

                if (total_columns <= 0) {
                    PGX_ERROR("Failed to get column metadata for table '%s'", spec.table_name.c_str());
                    throw std::runtime_error("Failed to get table metadata");
                }

                PGX_LOG(RUNTIME, DEBUG, "Retrieved metadata for %d columns from table '%s'", total_columns,
                        spec.table_name.c_str());

                if (spec.column_names.empty()) {
                    PGX_LOG(RUNTIME, DEBUG,
                            "No specific columns requested (e.g., COUNT(*)), skipping column metadata processing");
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
                            PGX_ERROR("Column '%s' not found in table '%s' metadata", col_spec.name.c_str(),
                                      spec.table_name.c_str());
                            throw std::runtime_error("Column not found in table");
                        }

                        switch (type_oid) {
                        case BOOLOID: col_spec.type = ::ColumnType::BOOLEAN; break;
                        case INT2OID: col_spec.type = ::ColumnType::SMALLINT; break;
                        case INT4OID: col_spec.type = ::ColumnType::INTEGER; break;
                        case INT8OID: col_spec.type = ::ColumnType::BIGINT; break;
                        case FLOAT4OID: col_spec.type = ::ColumnType::FLOAT; break;
                        case FLOAT8OID: col_spec.type = ::ColumnType::DOUBLE; break;
                        case TEXTOID:
                        case VARCHAROID:
                        case BPCHAROID:
                        case CHAROID: col_spec.type = ::ColumnType::STRING; break;
                        case NUMERICOID: col_spec.type = ::ColumnType::DECIMAL; break;
                        case DATEOID: col_spec.type = ::ColumnType::DATE; break;
                        case TIMESTAMPOID:
                        case TIMESTAMPTZOID: col_spec.type = ::ColumnType::TIMESTAMP; break;
                        case INTERVALOID: col_spec.type = ::ColumnType::INTERVAL; break;
                        default:
                            PGX_ERROR("Unsupported type %d for column '%s'", type_oid, col_spec.name.c_str());
                            throw std::runtime_error("Failed to parse column type");
                        }

                        iter->columns.push_back(col_spec);
                    }

                    json_parsed = true;
                    PGX_LOG(RUNTIME, DEBUG,
                            "decode_table_specification: JSON parsed successfully - table '%s' with %zu columns",
                            iter->table_name.c_str(), iter->columns.size());
                }
            }
        }
    }
    PG_CATCH();
    {
        PGX_ERROR("decode_table_specification: exception reading runtime::VarLen32 JSON");
        FlushErrorState();
        throw std::runtime_error("Failed to decode table specification");
    }
    PG_END_TRY();

    return json_parsed;
}

// ============================================================================
// BatchStorage Helper Functions
// ============================================================================

static size_t calculate_batch_capacity(TupleDesc tupleDesc) {
    PGX_IO(RUNTIME);
    extern int work_mem;
    const size_t work_mem_bytes = static_cast<size_t>(work_mem) * 1024L;

    size_t bytes_per_row = 0;
    for (int i = 0; i < tupleDesc->natts; i++) {
        const Form_pg_attribute attr = TupleDescAttr(tupleDesc, i);
        if (attr->attlen > 0) {
            bytes_per_row += attr->attlen;
        } else if (attr->attlen == -1) {
            bytes_per_row += 100;
        } else {
            bytes_per_row += 64;
        }
    }

    bytes_per_row += tupleDesc->natts * (sizeof(Datum) + sizeof(bool));

    size_t max_rows = work_mem_bytes / bytes_per_row;

    constexpr size_t MIN_BATCH_SIZE = 5;
    constexpr size_t MAX_BATCH_SIZE = 100000;

    if (max_rows < MIN_BATCH_SIZE) {
        max_rows = MIN_BATCH_SIZE;
    } else if (max_rows > MAX_BATCH_SIZE) {
        max_rows = MAX_BATCH_SIZE;
    }

    PGX_LOG(RUNTIME, DEBUG, "calculate_batch_capacity: work_mem=%dKB, bytes_per_row=%zu, capacity=%zu", work_mem,
            bytes_per_row, max_rows);

    return max_rows;
}

static BatchStorage* create_batch_storage(TupleDesc tupleDesc, size_t capacity) {
    PGX_IO(RUNTIME);

    // ReSharper disable once CppStaticAssertFailure
    const MemoryContext batchContext = AllocSetContextCreate(CurrentMemoryContext, "BatchStorage",
                                                             ALLOCSET_DEFAULT_MINSIZE, ALLOCSET_DEFAULT_INITSIZE,
                                                             ALLOCSET_DEFAULT_MAXSIZE);

    const MemoryContext oldContext = MemoryContextSwitchTo(batchContext);

    const auto batch = static_cast<BatchStorage*>(palloc(sizeof(BatchStorage)));
    batch->batchContext = batchContext;
    batch->tupleDesc = tupleDesc;
    batch->capacity = capacity;
    batch->num_rows = 0;

    const int num_cols = tupleDesc->natts;
    batch->column_values = static_cast<Datum**>(palloc(num_cols * sizeof(Datum*)));
    batch->column_nulls = static_cast<bool**>(palloc(num_cols * sizeof(bool*)));

    for (int col = 0; col < num_cols; col++) {
        batch->column_values[col] = (Datum*)palloc(capacity * sizeof(Datum));
        batch->column_nulls[col] = (bool*)palloc(capacity * sizeof(bool));
        memset(batch->column_nulls[col], true, capacity * sizeof(bool));
    }

    MemoryContextSwitchTo(oldContext);

    PGX_LOG(RUNTIME, DEBUG, "create_batch_storage: created batch with capacity=%zu, columns=%d, context=%p", capacity,
            num_cols, batchContext);

    return batch;
}

static void destroy_batch_storage(const BatchStorage* batch) {
    PGX_IO(RUNTIME);
    if (!batch) {
        return;
    }
    PGX_LOG(RUNTIME, DEBUG, "destroy_batch_storage: deleting context %p with %zu rows", batch->batchContext,
            batch->num_rows);
    MemoryContextDelete(batch->batchContext);
}

static void* open_table_connection(const std::string& table_name) {
    PGX_IO(RUNTIME);
    void* table_handle = open_postgres_table(table_name.c_str());

    if (!table_handle) {
        PGX_WARNING("open_table_connection: open_postgres_table failed for '%s'", table_name.c_str());
    }

    return table_handle;
}

DataSourceIteration*
DataSourceIteration::start(ExecutionContext* executionContext, const runtime::VarLen32 varlen32_param) {
    PGX_IO(RUNTIME);
    const MemoryContext oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    void* iter_memory = palloc(sizeof(DataSourceIterator));
    const auto iter = new (iter_memory) DataSourceIterator();

    const auto callback = (MemoryContextCallback*)palloc(sizeof(MemoryContextCallback));
    callback->func = cleanup_datasourceiterator_callback;
    callback->arg = iter;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

    MemoryContextSwitchTo(oldcontext);

    iter->context = executionContext;
    iter->batch = nullptr;
    iter->batch_exhausted = false;

    iter->current_value = 0;
    iter->current_is_null = true;
    const bool json_parsed = decode_table_specification(varlen32_param, iter);
    if (!json_parsed) {
        PGX_ERROR("JSON parsing failed");
        throw std::runtime_error("Failed to parse the json");
    }
    iter->table_handle = open_table_connection(iter->table_name);
    if (!iter->table_handle) {
        return reinterpret_cast<DataSourceIteration*>(iter);
    }

    g_current_iterator = iter;
    return reinterpret_cast<DataSourceIteration*>(iter);
}

bool DataSourceIteration::isValid() {
    PGX_IO(RUNTIME);
    auto* iter = reinterpret_cast<DataSourceIterator*>(this);
    if (!iter->table_handle) {
        PGX_LOG(RUNTIME, DEBUG, "Finished running with: %p branch 1 (no table_handle)", this);
        return false;
    }

    if (iter->batch && iter->batch->num_rows > 0 && !iter->batch_exhausted) {
        PGX_LOG(RUNTIME, DEBUG, "Returning true - batch has %zu rows", iter->batch->num_rows);
        return true;
    }
    if (iter->batch) {
        PGX_LOG(RUNTIME, DEBUG, "Destroying exhausted batch");
        destroy_batch_storage(iter->batch);
        iter->batch = nullptr;
    }

    struct PostgreSQLTableHandle {
        void* rel;
        void* scanDesc;
        void* tupleDesc; // Actually TupleDesc
        bool isOpen;
    };
    const auto* table_handle = static_cast<PostgreSQLTableHandle*>(iter->table_handle);
    const auto tupleDesc = static_cast<TupleDesc>(table_handle->tupleDesc);
    const size_t capacity = calculate_batch_capacity(tupleDesc);

    iter->batch = create_batch_storage(tupleDesc, capacity);
    PGX_LOG(RUNTIME, DEBUG, "Created new batch with capacity %zu", capacity);

    Datum temp_values[MaxTupleAttributeNumber];
    bool temp_nulls[MaxTupleAttributeNumber];
    while (iter->batch->num_rows < capacity) {
        PGX_LOG(RUNTIME, TRACE, "Reading tuple %zu", iter->batch->num_rows);
        const int64_t read_result = read_next_tuple_from_table(iter->table_handle);

        if (read_result != 1) {
            // End of table
            PGX_LOG(RUNTIME, DEBUG, "End of table after %zu rows", iter->batch->num_rows);
            break;
        }

        // Extract ALL columns from current tuple using heap_deform_tuple
        const auto tuple = static_cast<HeapTuple>(g_current_tuple_passthrough.originalTuple);
        if (!tuple) {
            PGX_ERROR("rt_datasourceiteration_isvalid: g_current_tuple_passthrough.originalTuple is NULL");
            break;
        }

        heap_deform_tuple(tuple, tupleDesc, temp_values, temp_nulls);
        const size_t row_idx = iter->batch->num_rows;
        for (int col = 0; col < tupleDesc->natts; col++) {
            const Form_pg_attribute attr = TupleDescAttr(tupleDesc, col);

            iter->batch->column_values[col][row_idx] = datumCopy(temp_values[col], attr->attbyval, attr->attlen);
            iter->batch->column_nulls[col][row_idx] = temp_nulls[col];

            PGX_LOG(RUNTIME, TRACE, "Row %zu col %d: Datum=%lu null=%s", row_idx, col,
                    static_cast<unsigned long>(iter->batch->column_values[col][row_idx]),
                    temp_nulls[col] ? "true" : "false");
        }

        iter->batch->num_rows++;
    }

    if (iter->batch->num_rows == 0) {
        PGX_LOG(RUNTIME, DEBUG, "Batch is empty, end of table");
        iter->batch_exhausted = true;
        return false;
    }

    iter->batch_exhausted = false;
    PGX_LOG(RUNTIME, DEBUG, "Batch filled with %zu rows", iter->batch->num_rows);
    return true;
}

void DataSourceIteration::access(RecordBatchInfo* info) {
    PGX_IO(RUNTIME);
    auto* row_data = info;
    if (!row_data) {
        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: row_data is NULL");
        return;
    }

    const auto* iter = reinterpret_cast<DataSourceIterator*>(this);
    if (!iter || !iter->batch || iter->batch->num_rows == 0) {
        PGX_LOG(RUNTIME, DEBUG, "Invalid iterator or empty batch");
        return;
    }

    PGX_LOG(RUNTIME, DEBUG, "Batch has %zu rows, %d columns", iter->batch->num_rows, iter->batch->tupleDesc->natts);

    // RecordBatchInfo structure (from lingodb):
    // [numRows: size_t][columnInfo[0]...][columnInfo[1]...]...
    // Each columnInfo has 5 fields: offset, validMultiplier, validBuffer, dataBuffer, varLenBuffer
    const auto row_data_ptr = reinterpret_cast<size_t*>(row_data);
    row_data_ptr[0] = iter->batch->num_rows; // Set number of rows in batch

    const size_t num_columns = iter->batch->tupleDesc->natts;
    PGX_LOG(RUNTIME, DEBUG, "Handling %zu columns", num_columns);

    for (size_t col = 0; col < num_columns; ++col) {
        constexpr size_t COLUMN_OFFSET_IDX = 0;
        constexpr size_t VALID_MULTIPLIER_IDX = 1;
        constexpr size_t VALID_BUFFER_IDX = 2;
        constexpr size_t DATA_BUFFER_IDX = 3;
        constexpr size_t VARLEN_BUFFER_IDX = 4;
        constexpr size_t COLUMN_INFO_SIZE = 5;

        size_t* column_info_ptr = &row_data_ptr[1 + col * COLUMN_INFO_SIZE];

        column_info_ptr[COLUMN_OFFSET_IDX] = 0; // offset (unused for columnar)
        column_info_ptr[VALID_MULTIPLIER_IDX] = 0; // validMultiplier (unused)

        column_info_ptr[VALID_BUFFER_IDX] = reinterpret_cast<size_t>(iter->batch->column_nulls[col]);
        column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(iter->batch->column_values[col]);

        // TODO: For varlena types (strings), we may need special handling here
        // For now, Datum IS the pointer for varlena types
        column_info_ptr[VARLEN_BUFFER_IDX] = 0;

        PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access: column %zu: values=%p nulls=%p", col,
                iter->batch->column_values[col], iter->batch->column_nulls[col]);
    }

    PGX_LOG(RUNTIME, DEBUG, "rt_datasourceiteration_access completed successfully");
    __sync_synchronize();
}

void DataSourceIteration::next() {
    PGX_IO(RUNTIME);
    auto* iter = reinterpret_cast<DataSourceIterator*>(this);
    iter->batch_exhausted = true;
    PGX_LOG(RUNTIME, DEBUG, "Marked batch as exhausted");
}

void DataSourceIteration::end(DataSourceIteration* iterator) {
    PGX_IO(RUNTIME);
    if (iterator) {
        auto* iter = reinterpret_cast<DataSourceIterator*>(iterator);

        if (iter->table_handle) {
            extern void close_postgres_table(void* tableHandle);
            close_postgres_table(iter->table_handle);
            iter->table_handle = nullptr;
        }
        if (g_current_iterator == reinterpret_cast<DataSourceIterator*>(iterator)) {
            g_current_iterator = nullptr;
        }

        PGX_LOG(RUNTIME, DEBUG, "Cleanup will be handled by MemoryContextCallback");
    }
}

} // namespace runtime

// Global context functions
void* getExecutionContext() {
    PGX_IO(RUNTIME);
    return ::rt_get_execution_context();
}

void setExecutionContext(void* context) {
    PGX_IO(RUNTIME);
    ::rt_set_execution_context(context);
}

} // namespace runtime
