#include "pgx-lower/runtime/PostgreSQLRuntime.h"
#include "pgx-lower/runtime/NumericConversion.h"
#include "lingodb/runtime/DataSourceIteration.h"
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

static void* g_execution_context = nullptr;

extern "C" {

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

    // Lingodb designed its string lookups to do this... so either we can make our storage work like this,
    // or we can edit the LLVM commands. Unfortunately, I opted to be lazy.
    int32_t** string_lengths;
    uint8_t*** string_data_ptrs;

    __int128** decimal_values;
};

struct DataSourceIterator {
    void* context;
    void* table_handle;

    std::string table_name;
    std::vector<ColumnSpec> columns;
    std::vector<int32_t> column_positions;

    BatchStorage* batch;
    size_t current_row_in_batch;

    int32_t current_id;
    bool current_id_is_null;
    int32_t current_col2;
    bool current_col2_is_null;
    int32_t current_value;
    bool current_is_null;
};

static DataSourceIterator* g_current_iterator = nullptr;

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

    PGX_ERROR("Failed to find column! %s %s", table_name.c_str(), column_name.c_str());
    throw std::runtime_error("Failed to find column!");
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

static void cleanup_tablebuilder_callback(void* arg) {
    PGX_IO(RUNTIME);
    // ReSharper disable once CppDeclaratorNeverUsed
    auto tb = static_cast<runtime::TableBuilder*>(arg);
}

namespace runtime {

TableBuilder::TableBuilder()
: data(nullptr)
, row_count(0)
, current_column_index(0)
, total_columns(0) {}

// ReSharper disable once CppParameterNeverUsed
TableBuilder* TableBuilder::create(VarLen32 schema_param) {
    PGX_IO(RUNTIME);

    const MemoryContext oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    void* builder_memory = palloc(sizeof(TableBuilder));
    const auto builder = new (builder_memory) TableBuilder();

    builder->total_columns = 0;
    PGX_LOG(RUNTIME, DEBUG, "Initialized with dynamic column tracking");

    const auto callback = static_cast<MemoryContextCallback*>(palloc(sizeof(MemoryContextCallback)));
    callback->func = cleanup_tablebuilder_callback;
    callback->arg = builder;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

    MemoryContextSwitchTo(oldcontext);
    return builder;
}

void TableBuilder::destroy(void* builder) {
    PGX_IO(RUNTIME);
    // Note: We don't need to do anything here because the MemoryContextCallback
    // will handle cleanup when the memory context is reset/deleted.
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

void TableBuilder::nextRow() {
    PGX_IO(RUNTIME);

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

void TableBuilder::addBool(const bool is_valid, const bool value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<bool>(this, is_valid, value);
}

void TableBuilder::addInt8(const bool is_valid, const int8_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int8_t>(this, is_valid, value);
}

void TableBuilder::addInt16(const bool is_valid, const int16_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int16_t>(this, is_valid, value);
}

void TableBuilder::addInt32(const bool is_valid, const int32_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int32_t>(this, is_valid, value);
}

void TableBuilder::addInt64(const bool is_valid, const int64_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, value);
}

void TableBuilder::addFloat32(const bool is_valid, const float value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<float>(this, is_valid, value);
}

void TableBuilder::addFloat64(const bool is_valid, const double value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<double>(this, is_valid, value);
}

void TableBuilder::addDecimal(const bool is_valid, const __int128 value) {
    PGX_IO(RUNTIME);

    if (!is_valid) {
        pgx_lower::runtime::table_builder_add_numeric(this, true, nullptr);
    } else {
        if (!this->next_decimal_scale.has_value()) {
            PGX_ERROR("Never set the decimal scale");
            throw std::runtime_error("Have no decimal scale");
        }
        const int32_t scale = this->next_decimal_scale.value();

        const Datum numeric_datum = i128_to_numeric(value, scale);
        const auto numeric_value = DatumGetNumeric(numeric_datum);

        PGX_LOG(RUNTIME, DEBUG, "addDecimal: created Numeric at %p (scale=%d, value=%lld)",
                numeric_value, scale, static_cast<long long>(value));

        pgx_lower::runtime::table_builder_add_numeric(this, false, numeric_value);
        this->next_decimal_scale = std::nullopt;
    }
}

void TableBuilder::addFixedSized(const bool is_valid, const int64_t value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int64_t>(this, is_valid, value);
}

void TableBuilder::addBinary(const bool is_valid, const VarLen32 value) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<VarLen32>(this, is_valid, value);
}

void TableBuilder::setNextDecimalScale(int32_t scale) {
    PGX_IO(RUNTIME);
    this->next_decimal_scale = scale;
}

static void cleanup_datasourceiterator_callback(void* arg) {
    PGX_IO(RUNTIME);
    if (const auto iter = static_cast<DataSourceIterator*>(arg)) {
        iter->~DataSourceIterator();
    }
}

// TODO: This function is uh... pretty gross. It should be returning iter, not a boolean. I also cannot be bothered
// fixing it now since it does its job and its just an abstracted away black box
static bool decode_table_specification(VarLen32 varlen32_param, DataSourceIterator* iter) {
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
            // ReSharper disable once CppUseStructuredBinding
            TableSpec spec = parse_table_spec(json_string.c_str());

            if (!spec.table_name.empty()) {
                size_t pipe_pos = spec.table_name.find("|oid:");
                if (pipe_pos != std::string::npos) {
                    std::string oid_str = spec.table_name.substr(pipe_pos + 5); // Skip "|oid:"
                    Oid table_oid = static_cast<Oid>(std::stoul(oid_str));
                    g_jit_table_oid = table_oid;
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

static size_t calculate_batch_capacity(const TupleDesc tupleDesc) {
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

static BatchStorage* create_batch_storage(const TupleDesc tupleDesc, const size_t num_cols, const size_t capacity) {
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

    batch->column_values = static_cast<Datum**>(palloc(num_cols * sizeof(Datum*)));
    batch->column_nulls = static_cast<bool**>(palloc(num_cols * sizeof(bool*)));
    batch->string_lengths = static_cast<int32_t**>(palloc(num_cols * sizeof(int32_t*)));
    batch->string_data_ptrs = static_cast<uint8_t***>(palloc(num_cols * sizeof(uint8_t**)));
    batch->decimal_values = static_cast<__int128**>(palloc(num_cols * sizeof(__int128*)));

    for (size_t col = 0; col < num_cols; col++) {
        batch->column_values[col] = static_cast<Datum*>(palloc(capacity * sizeof(Datum)));
        batch->column_nulls[col] = static_cast<bool*>(palloc(capacity * sizeof(bool)));
        memset(batch->column_nulls[col], true, capacity * sizeof(bool));

        batch->string_lengths[col] = static_cast<int32_t*>(palloc(capacity * sizeof(int32_t)));
        batch->string_data_ptrs[col] = static_cast<uint8_t**>(palloc(capacity * sizeof(uint8_t*)));
        memset(batch->string_lengths[col], 0, capacity * sizeof(int32_t));
        memset(batch->string_data_ptrs[col], 0, capacity * sizeof(uint8_t*));

        // __int128 requires 16-byte alignment. palloc() only guarantees 8-byte (MAXALIGN).
        const size_t alloc_size = capacity * sizeof(__int128) + 16;
        void* raw_ptr = palloc(alloc_size);
        const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_ptr);
        const uintptr_t aligned_addr = (raw_addr + 15) & ~static_cast<uintptr_t>(15);
        batch->decimal_values[col] = reinterpret_cast<__int128*>(aligned_addr);
        memset(batch->decimal_values[col], 0, capacity * sizeof(__int128));

        if ((reinterpret_cast<uintptr_t>(batch->decimal_values[col]) & 15) != 0) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                            errmsg("decimal_values[%zu] alignment failed: %p", col, batch->decimal_values[col])));
        }
    }

    MemoryContextSwitchTo(oldContext);

    PGX_LOG(RUNTIME, DEBUG, "Created batch with capacity=%zu, columns=%zu, context=%p", capacity, num_cols, batchContext);

    return batch;
}

static void destroy_batch_storage(const BatchStorage* batch) {
    PGX_IO(RUNTIME);
    if (!batch) {
        return;
    }
    PGX_LOG(RUNTIME, DEBUG, "Deleting context %p with %zu rows", batch->batchContext, batch->num_rows);
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

DataSourceIteration* DataSourceIteration::start(ExecutionContext* executionContext, const VarLen32 varlen32_param) {
    PGX_IO(RUNTIME);
    const MemoryContext oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    void* iter_memory = palloc(sizeof(DataSourceIterator));
    const auto iter = new (iter_memory) DataSourceIterator();

    const auto callback = static_cast<MemoryContextCallback*>(palloc(sizeof(MemoryContextCallback)));
    callback->func = cleanup_datasourceiterator_callback;
    callback->arg = iter;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, callback);

    MemoryContextSwitchTo(oldcontext);

    iter->context = executionContext;
    iter->batch = nullptr;
    iter->current_row_in_batch = 0;

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

    iter->column_positions.reserve(iter->columns.size());
    for (const auto& col_spec : iter->columns) {
        const int32_t pg_idx = get_column_position(iter->table_name, col_spec.name);
        iter->column_positions.push_back(pg_idx);
        PGX_LOG(RUNTIME, DEBUG, "Cached column '%s' at position %d", col_spec.name.c_str(), pg_idx);
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

    if (iter->batch && iter->current_row_in_batch < iter->batch->num_rows) {
        PGX_LOG(RUNTIME, DEBUG, "Returning true - current_row=%zu in batch with %zu rows", iter->current_row_in_batch,
                iter->batch->num_rows);
        return true;
    }
    if (iter->batch) {
        PGX_LOG(RUNTIME, DEBUG, "Destroying exhausted batch (had %zu rows)", iter->batch->num_rows);
        destroy_batch_storage(iter->batch);
        iter->batch = nullptr;
        iter->current_row_in_batch = 0;
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

    const size_t num_cols = iter->columns.size(); // Use JSON column count
    iter->batch = create_batch_storage(tupleDesc, num_cols, capacity);
    PGX_LOG(RUNTIME, DEBUG, "Created new batch with capacity %zu, JSON columns %zu", capacity, num_cols);

    Datum temp_values[MaxTupleAttributeNumber];
    bool temp_nulls[MaxTupleAttributeNumber];
    while (iter->batch->num_rows < capacity) {
        PGX_HOT_LOG(RUNTIME, TRACE, "Reading tuple %zu", iter->batch->num_rows);
        const int64_t read_result = read_next_tuple_from_table(iter->table_handle);

        if (read_result != 1) {
            PGX_LOG(RUNTIME, DEBUG, "End of table after %zu rows", iter->batch->num_rows);
            break;
        }

        const auto tuple = g_current_tuple_passthrough.originalTuple;
        if (!tuple) {
            PGX_ERROR("g_current_tuple_passthrough.originalTuple is NULL");
            break;
        }

        heap_deform_tuple(tuple, tupleDesc, temp_values, temp_nulls);
        const size_t row_idx = iter->batch->num_rows;

        for (size_t json_col_idx = 0; json_col_idx < iter->columns.size(); json_col_idx++) {
            // ReSharper disable once CppUseStructuredBinding
            const auto& col_spec = iter->columns[json_col_idx];
            const int pg_col_idx = iter->column_positions[json_col_idx];

            if (pg_col_idx < 0 || pg_col_idx >= tupleDesc->natts) {
                PGX_ERROR("Column %s not found in table %s", col_spec.name.c_str(), iter->table_name.c_str());
                iter->batch->column_nulls[json_col_idx][row_idx] = true;
                continue;
            }

            const Form_pg_attribute attr = TupleDescAttr(tupleDesc, pg_col_idx);

            if (col_spec.type == ::ColumnType::STRING) {
                if (temp_nulls[pg_col_idx]) {
                    iter->batch->string_lengths[json_col_idx][row_idx] = 0;
                    iter->batch->string_data_ptrs[json_col_idx][row_idx] = nullptr;
                } else {
                    const Datum transferred_datum = datumTransfer(temp_values[pg_col_idx], attr->attbyval, attr->attlen);
                    const auto pg_text = DatumGetTextPP(transferred_datum);
                    const char* str_data = VARDATA_ANY(pg_text);
                    const int str_len = VARSIZE_ANY_EXHDR(pg_text);

                    iter->batch->string_lengths[json_col_idx][row_idx] = str_len;
                    iter->batch->string_data_ptrs[json_col_idx][row_idx] = reinterpret_cast<uint8_t*>(const_cast<char*>(str_data));
                    iter->batch->column_values[json_col_idx][row_idx] = transferred_datum;

                    PGX_HOT_LOG(RUNTIME, TRACE, "Row %zu: col[%zu]='%s' STRING: len=%d, data=%p", row_idx, json_col_idx,
                            col_spec.name.c_str(), str_len, str_data);
                }
            } else if (attr->atttypid == NUMERICOID) {
                if (temp_nulls[pg_col_idx]) {
                    iter->batch->decimal_values[json_col_idx][row_idx] = 0;
                    PGX_HOT_LOG(RUNTIME, DEBUG, "Row %zu: col[%zu]='%s' NUMERIC is NULL",
                            row_idx, json_col_idx, col_spec.name.c_str());
                } else {
                    int32_t type_scale = 0;
                    if (attr->atttypmod >= 0) {
                        // typmod encoding: ((precision << 16) | scale) + 4
                        const int32_t adjusted = attr->atttypmod - 4; // VARHDRSZ = 4
                        type_scale = adjusted & 0xFFFF;
                    } else {
                        type_scale = 6;
                    }

                    __int128 scaled_value = numeric_to_i128(temp_values[pg_col_idx], type_scale);

                    iter->batch->decimal_values[json_col_idx][row_idx] = scaled_value;
                    PGX_HOT_LOG(RUNTIME, DEBUG, "Row %zu: col[%zu]='%s' NUMERIC->i128: scale=%d, value=%lld",
                            row_idx, json_col_idx, col_spec.name.c_str(), type_scale,
                            static_cast<long long>(scaled_value));
                }
            } else if (attr->atttypid == INTERVALOID) {
                if (temp_nulls[pg_col_idx]) {
                    iter->batch->column_values[json_col_idx][row_idx] = 0;
                    iter->batch->column_nulls[json_col_idx][row_idx] = false;
                    PGX_HOT_LOG(RUNTIME, DEBUG, "Row %zu: col[%zu]='%s' INTERVAL is NULL",
                            row_idx, json_col_idx, col_spec.name.c_str());
                } else {
                    // Extract full interval and convert to microseconds (LingoDB's !db.interval<daytime> format)
                    Interval* interval = DatumGetIntervalP(temp_values[pg_col_idx]);

                    // Combine time (microseconds) + day (convert to microseconds) + month (approximate)
                    int64_t totalMicroseconds = interval->time +
                        (static_cast<int64_t>(interval->day) * USECS_PER_DAY);

                    if (interval->month != 0) {
                        // TODO Phase N+: This loses precision for month-based intervals
                        // Using 30-day approximation (matches frontend/SQL translation constants)
                        constexpr int64_t AVERAGE_DAYS_PER_MONTH = 30;
                        int64_t monthMicroseconds = static_cast<int64_t>(
                            interval->month * AVERAGE_DAYS_PER_MONTH * USECS_PER_DAY);
                        totalMicroseconds += monthMicroseconds;
                    }

                    // Store as int64 Datum (LingoDB format)
                    iter->batch->column_values[json_col_idx][row_idx] = Int64GetDatum(totalMicroseconds);

                    PGX_HOT_LOG(RUNTIME, DEBUG, "Row %zu: col[%zu]='%s' INTERVAL: time=%lld, day=%d, month=%d → total_us=%lld",
                            row_idx, json_col_idx, col_spec.name.c_str(),
                            static_cast<long long>(interval->time), interval->day, interval->month,
                            static_cast<long long>(totalMicroseconds));
                }
            } else {
                if (temp_nulls[pg_col_idx]) {
                    iter->batch->column_values[json_col_idx][row_idx] = 0;
                    PGX_HOT_LOG(RUNTIME, TRACE, "Row %zu: JSON_col[%zu]='%s' from PG_col[%d] is NULL", row_idx,
                            json_col_idx, col_spec.name.c_str(), pg_col_idx);
                } else {
                    iter->batch->column_values[json_col_idx][row_idx] = datumTransfer(temp_values[pg_col_idx], attr->attbyval,
                                                                                  attr->attlen);

                    PGX_HOT_LOG(RUNTIME, TRACE, "Row %zu: JSON_col[%zu]='%s' from PG_col[%d] Datum=%lu", row_idx,
                            json_col_idx, col_spec.name.c_str(), pg_col_idx,
                            static_cast<unsigned long>(iter->batch->column_values[json_col_idx][row_idx]));
                }
            }

            // A bit goofy - requires inverting because lingodb stores the opposite null flags...
            iter->batch->column_nulls[json_col_idx][row_idx] = !temp_nulls[pg_col_idx];
        }

        iter->batch->num_rows++;
    }

    if (iter->batch->num_rows == 0) {
        PGX_LOG(RUNTIME, DEBUG, "Batch is empty, end of table");
        return false;
    }

    iter->current_row_in_batch = 0;
    PGX_LOG(RUNTIME, DEBUG, "Batch filled with %zu rows, current_row reset to 0", iter->batch->num_rows);
    return true;
}

void DataSourceIteration::access(RecordBatchInfo* info) {
    PGX_IO(RUNTIME);
    auto* row_data = info;
    if (!row_data) {
        PGX_LOG(RUNTIME, DEBUG, "row_data is NULL");
        return;
    }

    const auto* iter = reinterpret_cast<DataSourceIterator*>(this);
    if (!iter->batch || iter->current_row_in_batch >= iter->batch->num_rows) {
        PGX_LOG(RUNTIME, DEBUG, "Invalid iterator, empty batch, or current_row out of range");
        return;
    }

    const size_t row_idx = iter->current_row_in_batch;
    const size_t num_columns = iter->columns.size();

    PGX_LOG(RUNTIME, DEBUG, "Accessing row %zu/%zu from batch (batch has %zu JSON columns)", row_idx,
            iter->batch->num_rows, num_columns);

    // RecordBatchInfo structure (from lingodb):
    // [numRows: size_t][columnInfo[0]...][columnInfo[1]...]...
    // Each columnInfo has 5 fields: offset, validMultiplier, validBuffer, dataBuffer, varLenBuffer
    const auto row_data_ptr = reinterpret_cast<size_t*>(row_data);
    row_data_ptr[0] = 1;

    for (size_t col = 0; col < num_columns; ++col) {
        constexpr size_t COLUMN_OFFSET_IDX = 0;
        constexpr size_t VALID_MULTIPLIER_IDX = 1;
        constexpr size_t VALID_BUFFER_IDX = 2;
        constexpr size_t DATA_BUFFER_IDX = 3;
        constexpr size_t VARLEN_BUFFER_IDX = 4;
        constexpr size_t COLUMN_INFO_SIZE = 5;

        size_t* column_info_ptr = &row_data_ptr[1 + col * COLUMN_INFO_SIZE];

        column_info_ptr[COLUMN_OFFSET_IDX] = 0;
        column_info_ptr[VALID_MULTIPLIER_IDX] = 0; // validMultiplier (unused)

        column_info_ptr[VALID_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->column_nulls[col][row_idx]);

        // For string columns, pass Arrow format pointers (length and data pointer)
        // For simple types, Datum contains the value directly
        if (iter->columns[col].type == ::ColumnType::STRING) {
            // Arrow format: DATA_BUFFER points to int32_t length, VARLEN_BUFFER points to uint8_t* data pointer
            column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->string_lengths[col][row_idx]);
            column_info_ptr[VARLEN_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->string_data_ptrs[col][row_idx]);

            PGX_LOG(RUNTIME, TRACE, "access() col=%zu STRING (Arrow format):", col);
            PGX_LOG(RUNTIME, TRACE, "  Length: %d (at %p)", iter->batch->string_lengths[col][row_idx],
                    &iter->batch->string_lengths[col][row_idx]);
            PGX_LOG(RUNTIME, TRACE, "  Data pointer: %p (at %p)", iter->batch->string_data_ptrs[col][row_idx],
                    &iter->batch->string_data_ptrs[col][row_idx]);
            PGX_LOG(RUNTIME, TRACE, "  DATA_BUFFER_IDX → %p", reinterpret_cast<void*>(column_info_ptr[DATA_BUFFER_IDX]));
            PGX_LOG(RUNTIME, TRACE, "  VARLEN_BUFFER_IDX → %p",
                    reinterpret_cast<void*>(column_info_ptr[VARLEN_BUFFER_IDX]));
        } else if (iter->columns[col].type == ::ColumnType::DECIMAL) {
            column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->decimal_values[col][row_idx]);
            column_info_ptr[VARLEN_BUFFER_IDX] = 0;

            PGX_LOG(RUNTIME, DEBUG, "access() col=%zu DECIMAL i128 at %p, value=%lld", col,
                    &iter->batch->decimal_values[col][row_idx],
                    static_cast<long long>(iter->batch->decimal_values[col][row_idx]));
        } else {
            // Pass address of Datum itself (contains value)
            column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->column_values[col][row_idx]);
            column_info_ptr[VARLEN_BUFFER_IDX] = 0;
        }
    }

    __sync_synchronize();
}

void DataSourceIteration::next() {
    PGX_IO(RUNTIME);
    auto* iter = reinterpret_cast<DataSourceIterator*>(this);
    if (!iter->batch) {
        PGX_LOG(RUNTIME, DEBUG, "next() called with no batch");
        return;
    }

    PGX_LOG(RUNTIME, DEBUG, "next(): advancing from row %zu to %zu (batch has %zu rows)", iter->current_row_in_batch,
            iter->current_row_in_batch + 1, iter->batch->num_rows);

    iter->current_row_in_batch++;
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
    return rt_get_execution_context();
}

void setExecutionContext(void* context) {
    PGX_IO(RUNTIME);
    rt_set_execution_context(context);
}

} // namespace runtime
