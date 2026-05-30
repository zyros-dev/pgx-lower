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


extern void store_bool_result(int32_t column_index, bool value, bool is_null);





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

// Per-column decode metadata cached at iterator-start time so the per-row hot
// loop in process_tuple_into_batch avoids TupleDescAttr lookups, type-OID
// re-checks, and atttypmod→scale arithmetic on every tuple. See spec 05.
enum class DecodeKind : uint8_t {
    STRING, // VARDATA_ANY + length, datumTransfer
    NUMERIC, // numeric_to_i128 with cached scale
    INTERVAL, // Interval struct → microseconds
    DATUM_BYVAL, // pass-through Datum, no copy
    DATUM_BYREF, // datumTransfer to batch context
};

struct ColumnDecodeMeta {
    DecodeKind kind;
    bool attbyval;
    int16 attlen;
    int32_t numeric_scale; // only meaningful when kind == NUMERIC
};

struct BatchStorage {
    MemoryContext batch_context;
    TupleDesc tuple_desc;

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
    std::vector<ColumnDecodeMeta> column_decode_meta;

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
    const int32_t ATTNUM = get_column_attnum(table_name.c_str(), column_name.c_str());
    if (ATTNUM > 0) {
        // Convert from 1-based PostgreSQL attnum to 0-based index
        return ATTNUM - 1;
    }

    PGX_ERROR("Failed to find column! %s %s", table_name.c_str(), column_name.c_str());
    throw std::runtime_error("Failed to find column!");
}

static TableSpec parse_table_spec(const char* json_str) {
    PGX_IO(RUNTIME);
    TableSpec spec;

    try {
        PGX_LOG(RUNTIME, DEBUG, "parse_table_spec: parsing JSON: %s", json_str);

        using Json = nlohmann::json;
        Json j = Json::parse(json_str);

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
    auto *tb = static_cast<runtime::TableBuilder*>(arg);
}

namespace runtime {

TableBuilder::TableBuilder()
: data(nullptr)
, row_count(0)
, current_column_index(0)
, total_columns(0) {}

// ReSharper disable once CppParameterNeverUsed
TableBuilder* TableBuilder::create(VarLen32  /*schema_param*/) {
    PGX_IO(RUNTIME);

    const MemoryContext OLDCONTEXT = MemoryContextSwitchTo(CurrentMemoryContext);

    void* const builder_memory = palloc(sizeof(TableBuilder));
    auto *const BUILDER = new (builder_memory) TableBuilder();

    BUILDER->total_columns = 0;
    PGX_LOG(RUNTIME, DEBUG, "Initialized with dynamic column tracking");

    auto *const CALLBACK = static_cast<MemoryContextCallback*>(palloc(sizeof(MemoryContextCallback)));
    CALLBACK->func = cleanup_tablebuilder_callback;
    CALLBACK->arg = BUILDER;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, CALLBACK);

    MemoryContextSwitchTo(OLDCONTEXT);
    return BUILDER;
}

void TableBuilder::destroy(void*  /*builder*/) {
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

void TableBuilder::addBool(const bool IS_VALID, const bool VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<bool>(this, IS_VALID, VALUE);
}

void TableBuilder::addInt8(const bool IS_VALID, const int8_t VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int8_t>(this, IS_VALID, VALUE);
}

void TableBuilder::addInt16(const bool IS_VALID, const int16_t VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int16_t>(this, IS_VALID, VALUE);
}

void TableBuilder::addInt32(const bool IS_VALID, const int32_t VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int32_t>(this, IS_VALID, VALUE);
}

void TableBuilder::addInt64(const bool IS_VALID, const int64_t VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int64_t>(this, IS_VALID, VALUE);
}

void TableBuilder::addFloat32(const bool IS_VALID, const float VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<float>(this, IS_VALID, VALUE);
}

void TableBuilder::addFloat64(const bool IS_VALID, const double VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<double>(this, IS_VALID, VALUE);
}

void TableBuilder::addDecimal(const bool IS_VALID, const __int128 VALUE) {
    PGX_IO(RUNTIME);

    if (!IS_VALID) {
        pgx_lower::runtime::table_builder_add_numeric(this, true, nullptr);
    } else {
        if (!this->next_decimal_scale.has_value()) {
            PGX_ERROR("Never set the decimal scale");
            throw std::runtime_error("Have no decimal scale");
        }
        const int32_t SCALE = this->next_decimal_scale.value();

        const Datum NUMERIC_DATUM = i128_to_numeric(VALUE, SCALE);
        auto *const NUMERIC_VALUE = DatumGetNumeric(NUMERIC_DATUM);

        PGX_LOG(RUNTIME, DEBUG, "addDecimal: created Numeric at %p (scale=%d, value=%lld)",
                NUMERIC_VALUE, SCALE, static_cast<long long>(VALUE));

        pgx_lower::runtime::table_builder_add_numeric(this, false, NUMERIC_VALUE);
        this->next_decimal_scale = std::nullopt;
    }
}

void TableBuilder::addFixedSized(const bool IS_VALID, const int64_t VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<int64_t>(this, IS_VALID, VALUE);
}

void TableBuilder::addBinary(const bool IS_VALID, const VarLen32 VALUE) {
    PGX_IO(RUNTIME);
    pgx_lower::runtime::table_builder_add<VarLen32>(this, IS_VALID, VALUE);
}

void TableBuilder::setNextDecimalScale(int32_t scale) {
    PGX_IO(RUNTIME);
    this->next_decimal_scale = scale;
}

static void cleanup_datasourceiterator_callback(void* arg) {
    PGX_IO(RUNTIME);
    if (auto *const ITER = static_cast<DataSourceIterator*>(arg)) {
        ITER->~DataSourceIterator();
    }
}

// TODO: This function is uh... pretty gross. It should be returning iter, not a boolean. I also cannot be bothered
// fixing it now since it does its job and its just an abstracted away black box
static bool decode_table_specification(VarLen32 varlen32_param, DataSourceIterator* iter) {
    PGX_IO(RUNTIME);
    uint32_t const actual_len = varlen32_param.getLen();
    const char* const json_spec = varlen32_param.data();

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
                size_t const pipe_pos = spec.table_name.find("|oid:");
                if (pipe_pos != std::string::npos) {
                    std::string const oid_str = spec.table_name.substr(pipe_pos + 5); // Skip "|oid:"
                    Oid const table_oid = static_cast<Oid>(std::stoul(oid_str));
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
                int32_t const total_columns = get_all_column_metadata(spec.table_name.c_str(), metadata,
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

static size_t calculate_batch_capacity(const TupleDesc TUPLE_DESC) {
    PGX_IO(RUNTIME);
    extern int work_mem;
    const size_t WORK_MEM_BYTES = static_cast<size_t>(work_mem) * 1024L;

    size_t bytes_per_row = 0;
    for (int i = 0; i < TUPLE_DESC->natts; i++) {
        const Form_pg_attribute ATTR = TupleDescAttr(TUPLE_DESC, i);
        if (ATTR->attlen > 0) {
            bytes_per_row += ATTR->attlen;
        } else if (ATTR->attlen == -1) {
            bytes_per_row += 100;
        } else {
            bytes_per_row += 64;
        }
    }

    bytes_per_row += TUPLE_DESC->natts * (sizeof(Datum) + sizeof(bool));

    size_t max_rows = WORK_MEM_BYTES / bytes_per_row;

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

static BatchStorage* create_batch_storage(const TupleDesc TUPLE_DESC, const size_t NUM_COLS, const size_t CAPACITY) {
    PGX_IO(RUNTIME);

    // ReSharper disable once CppStaticAssertFailure
    const MemoryContext BATCH_CONTEXT = AllocSetContextCreate(CurrentMemoryContext, "BatchStorage",
                                                             ALLOCSET_DEFAULT_MINSIZE, ALLOCSET_DEFAULT_INITSIZE,
                                                             ALLOCSET_DEFAULT_MAXSIZE);

    const MemoryContext OLD_CONTEXT = MemoryContextSwitchTo(BATCH_CONTEXT);

    auto *const BATCH = static_cast<BatchStorage*>(palloc(sizeof(BatchStorage)));
    BATCH->batch_context = BATCH_CONTEXT;
    BATCH->tuple_desc = TUPLE_DESC;
    BATCH->capacity = CAPACITY;
    BATCH->num_rows = 0;

    BATCH->column_values = static_cast<Datum**>(palloc(NUM_COLS * sizeof(Datum*)));
    BATCH->column_nulls = static_cast<bool**>(palloc(NUM_COLS * sizeof(bool*)));
    BATCH->string_lengths = static_cast<int32_t**>(palloc(NUM_COLS * sizeof(int32_t*)));
    BATCH->string_data_ptrs = static_cast<uint8_t***>(palloc(NUM_COLS * sizeof(uint8_t**)));
    BATCH->decimal_values = static_cast<__int128**>(palloc(NUM_COLS * sizeof(__int128*)));

    for (size_t col = 0; col < NUM_COLS; col++) {
        BATCH->column_values[col] = static_cast<Datum*>(palloc(CAPACITY * sizeof(Datum)));
        BATCH->column_nulls[col] = static_cast<bool*>(palloc(CAPACITY * sizeof(bool)));
        memset(BATCH->column_nulls[col], 1, CAPACITY * sizeof(bool));

        BATCH->string_lengths[col] = static_cast<int32_t*>(palloc(CAPACITY * sizeof(int32_t)));
        BATCH->string_data_ptrs[col] = static_cast<uint8_t**>(palloc(CAPACITY * sizeof(uint8_t*)));
        memset(BATCH->string_lengths[col], 0, CAPACITY * sizeof(int32_t));
        memset(BATCH->string_data_ptrs[col], 0, CAPACITY * sizeof(uint8_t*));

        // __int128 requires 16-byte alignment. palloc() only guarantees 8-byte (MAXALIGN).
        const size_t ALLOC_SIZE = CAPACITY * sizeof(__int128) + 16;
        void* const raw_ptr = palloc(ALLOC_SIZE);
        const uintptr_t RAW_ADDR = reinterpret_cast<uintptr_t>(raw_ptr);
        const uintptr_t ALIGNED_ADDR = (RAW_ADDR + 15) & ~static_cast<uintptr_t>(15);
        BATCH->decimal_values[col] = reinterpret_cast<__int128*>(ALIGNED_ADDR);
        memset(BATCH->decimal_values[col], 0, CAPACITY * sizeof(__int128));

        if ((reinterpret_cast<uintptr_t>(BATCH->decimal_values[col]) & 15) != 0) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                            errmsg("decimal_values[%zu] alignment failed: %p", col, BATCH->decimal_values[col])));
        }
    }

    MemoryContextSwitchTo(OLD_CONTEXT);

    PGX_LOG(RUNTIME, DEBUG, "Created batch with capacity=%zu, columns=%zu, context=%p", CAPACITY, NUM_COLS, BATCH_CONTEXT);

    return BATCH;
}

static void destroy_batch_storage(const BatchStorage* batch) {
    PGX_IO(RUNTIME);
    if (!batch) {
        return;
    }
    PGX_LOG(RUNTIME, DEBUG, "Deleting context %p with %zu rows", batch->batch_context, batch->num_rows);
    MemoryContextDelete(batch->batch_context);
}

static void* open_table_connection(const std::string& table_name) {
    PGX_IO(RUNTIME);
    void* const table_handle = open_postgres_table(table_name.c_str());

    if (!table_handle) {
        PGX_WARNING("open_table_connection: open_postgres_table failed for '%s'", table_name.c_str());
    }

    return table_handle;
}

DataSourceIteration* DataSourceIteration::start(ExecutionContext* execution_context, const VarLen32 VARLEN32_PARAM) {
    PGX_IO(RUNTIME);
    const MemoryContext OLDCONTEXT = MemoryContextSwitchTo(CurrentMemoryContext);

    void* const iter_memory = palloc(sizeof(DataSourceIterator));
    auto *const ITER = new (iter_memory) DataSourceIterator();

    auto *const CALLBACK = static_cast<MemoryContextCallback*>(palloc(sizeof(MemoryContextCallback)));
    CALLBACK->func = cleanup_datasourceiterator_callback;
    CALLBACK->arg = ITER;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, CALLBACK);

    MemoryContextSwitchTo(OLDCONTEXT);

    ITER->context = execution_context;
    ITER->batch = nullptr;
    ITER->current_row_in_batch = 0;

    ITER->current_value = 0;
    ITER->current_is_null = true;
    const bool JSON_PARSED = decode_table_specification(VARLEN32_PARAM, ITER);
    if (!JSON_PARSED) {
        PGX_ERROR("JSON parsing failed");
        throw std::runtime_error("Failed to parse the json");
    }
    ITER->table_handle = open_table_connection(ITER->table_name);
    if (!ITER->table_handle) {
        return reinterpret_cast<DataSourceIteration*>(ITER);
    }

    ITER->column_positions.reserve(ITER->columns.size());
    for (const auto& col_spec : ITER->columns) {
        const int32_t PG_IDX = get_column_position(ITER->table_name, col_spec.name);
        ITER->column_positions.push_back(PG_IDX);
        PGX_LOG(RUNTIME, DEBUG, "Cached column '%s' at position %d", col_spec.name.c_str(), PG_IDX);
    }

    // Pre-resolve per-column decode metadata once. The per-row hot loop in
    // process_tuple_into_batch reads from this vector instead of doing a
    // TupleDescAttr lookup + type-OID re-dispatch + atttypmod→scale on every
    // tuple.
    {
        const TupleDesc TUPLE_DESC = get_table_handle_tupledesc(ITER->table_handle);
        ITER->column_decode_meta.reserve(ITER->columns.size());
        for (size_t i = 0; i < ITER->columns.size(); i++) {
            ColumnDecodeMeta meta{};
            const int32_t PG_IDX = ITER->column_positions[i];
            if (PG_IDX < 0 || !TUPLE_DESC || PG_IDX >= TUPLE_DESC->natts) {
                meta.kind = DecodeKind::DATUM_BYVAL;
                ITER->column_decode_meta.push_back(meta);
                continue;
            }
            const Form_pg_attribute ATTR = TupleDescAttr(TUPLE_DESC, PG_IDX);
            meta.attbyval = ATTR->attbyval;
            meta.attlen = ATTR->attlen;
            if (ITER->columns[i].type == ::ColumnType::STRING) {
                meta.kind = DecodeKind::STRING;
            } else if (ATTR->atttypid == NUMERICOID) {
                meta.kind = DecodeKind::NUMERIC;
                if (ATTR->atttypmod >= 0) {
                    meta.numeric_scale = (ATTR->atttypmod - 4) & 0xFFFF;
                } else {
                    meta.numeric_scale = 6;
                }
            } else if (ATTR->atttypid == INTERVALOID) {
                meta.kind = DecodeKind::INTERVAL;
            } else {
                meta.kind = ATTR->attbyval ? DecodeKind::DATUM_BYVAL : DecodeKind::DATUM_BYREF;
            }
            ITER->column_decode_meta.push_back(meta);
        }
    }

    g_current_iterator = ITER;
    return reinterpret_cast<DataSourceIteration*>(ITER);
}

namespace {
    [[nodiscard]] bool check_batch_validity(DataSourceIterator* iter) noexcept {
        if (!iter->table_handle) {
            PGX_LOG(RUNTIME, DEBUG, "Finished running with: %p branch 1 (no table_handle)", iter);
            return false;
        }

        if (iter->batch && iter->current_row_in_batch < iter->batch->num_rows) {
            PGX_LOG(RUNTIME, DEBUG, "Returning true - current_row=%zu in batch with %zu rows",
                    iter->current_row_in_batch, iter->batch->num_rows);
            return true;
        }

        return false;
    }

    void prepare_new_batch(DataSourceIterator* iter, TupleDesc tuple_desc) {
        if (iter->batch) {
            PGX_LOG(RUNTIME, DEBUG, "Destroying exhausted batch (had %zu rows)", iter->batch->num_rows);
            destroy_batch_storage(iter->batch);
            iter->batch = nullptr;
            iter->current_row_in_batch = 0;
        }

        const size_t CAPACITY = calculate_batch_capacity(tuple_desc);
        const size_t NUM_COLS = iter->columns.size();
        iter->batch = create_batch_storage(tuple_desc, NUM_COLS, CAPACITY);
        PGX_LOG(RUNTIME, DEBUG, "Created new batch with capacity %zu, JSON columns %zu", CAPACITY, NUM_COLS);
    }

    void process_tuple_into_batch(DataSourceIterator* iter, TupleDesc tuple_desc,
                                   Datum* temp_values, bool* temp_nulls) {
        auto *const TUPLE = g_current_tuple_passthrough.originalTuple;
        if (!TUPLE) {
            PGX_ERROR("g_current_tuple_passthrough.originalTuple is NULL");
            return;
        }

        heap_deform_tuple(TUPLE, tuple_desc, temp_values, temp_nulls);
        const size_t ROW_IDX = iter->batch->num_rows;
        const size_t NUM_COLS = iter->columns.size();
        const ColumnDecodeMeta* const metas = iter->column_decode_meta.data();
        const int32_t* const positions = iter->column_positions.data();

        for (size_t json_col_idx = 0; json_col_idx < NUM_COLS; json_col_idx++) {
            const ColumnDecodeMeta& meta = metas[json_col_idx];
            const int PG_COL_IDX = positions[json_col_idx];
            const bool IS_NULL = temp_nulls[PG_COL_IDX];
            const Datum VALUE = temp_values[PG_COL_IDX];

            switch (meta.kind) {
            case DecodeKind::STRING: {
                if (IS_NULL) {
                    iter->batch->string_lengths[json_col_idx][ROW_IDX] = 0;
                    iter->batch->string_data_ptrs[json_col_idx][ROW_IDX] = nullptr;
                } else {
                    const Datum TRANSFERRED_DATUM = datumTransfer(VALUE, meta.attbyval, meta.attlen);
                    auto *const PG_TEXT = DatumGetTextPP(TRANSFERRED_DATUM);
                    iter->batch->string_lengths[json_col_idx][ROW_IDX] = VARSIZE_ANY_EXHDR(PG_TEXT);
                    iter->batch->string_data_ptrs[json_col_idx][ROW_IDX] = reinterpret_cast<uint8_t*>(
                        const_cast<char*>(VARDATA_ANY(PG_TEXT)));
                    iter->batch->column_values[json_col_idx][ROW_IDX] = TRANSFERRED_DATUM;
                }
                break;
            }
            case DecodeKind::NUMERIC: {
                iter->batch->decimal_values[json_col_idx][ROW_IDX] = IS_NULL
                                                                         ? __int128{0}
                                                                         : numeric_to_i128(VALUE, meta.numeric_scale);
                break;
            }
            case DecodeKind::INTERVAL: {
                if (IS_NULL) {
                    iter->batch->column_values[json_col_idx][ROW_IDX] = 0;
                    iter->batch->column_nulls[json_col_idx][ROW_IDX] = false;
                } else {
                    Interval* const interval = DatumGetIntervalP(VALUE);
                    int64_t total_microseconds = interval->time +
                        (static_cast<int64_t>(interval->day) * USECS_PER_DAY);
                    if (interval->month != 0) {
                        constexpr int64_t AVERAGE_DAYS_PER_MONTH = 30;
                        total_microseconds += static_cast<int64_t>(interval->month) * AVERAGE_DAYS_PER_MONTH
                                             * USECS_PER_DAY;
                    }
                    iter->batch->column_values[json_col_idx][ROW_IDX] = Int64GetDatum(total_microseconds);
                }
                break;
            }
            case DecodeKind::DATUM_BYVAL: {
                iter->batch->column_values[json_col_idx][ROW_IDX] = IS_NULL ? Datum{0} : VALUE;
                break;
            }
            case DecodeKind::DATUM_BYREF: {
                iter->batch->column_values[json_col_idx][ROW_IDX] = IS_NULL ? Datum{0}
                                                                            : datumTransfer(VALUE, meta.attbyval,
                                                                                            meta.attlen);
                break;
            }
            }

            iter->batch->column_nulls[json_col_idx][ROW_IDX] = !IS_NULL;
        }

        iter->batch->num_rows++;
    }

    [[nodiscard]] bool read_and_fill_batch(DataSourceIterator* iter, TupleDesc tuple_desc) {
        Datum temp_values[MaxTupleAttributeNumber];
        bool temp_nulls[MaxTupleAttributeNumber];
        const size_t CAPACITY = iter->batch->capacity;

        while (iter->batch->num_rows < CAPACITY) {
            PGX_HOT_LOG(RUNTIME, TRACE, "Reading tuple %zu", iter->batch->num_rows);
            const int64_t READ_RESULT = read_next_tuple_from_table(iter->table_handle);

            if (READ_RESULT != 1) {
                PGX_LOG(RUNTIME, DEBUG, "End of table after %zu rows", iter->batch->num_rows);
                break;
            }

            process_tuple_into_batch(iter, tuple_desc, temp_values, temp_nulls);
        }

        return iter->batch->num_rows > 0;
    }

    [[nodiscard]] bool finalize_batch(DataSourceIterator* iter) noexcept {
        if (iter->batch->num_rows == 0) {
            PGX_LOG(RUNTIME, DEBUG, "Batch is empty, end of table");
            return false;
        }

        iter->current_row_in_batch = 0;
        PGX_LOG(RUNTIME, DEBUG, "Batch filled with %zu rows, current_row reset to 0", iter->batch->num_rows);
        return true;
    }
} // anonymous namespace

bool DataSourceIteration::isValid() {
    PGX_IO(RUNTIME);
    auto* iter = reinterpret_cast<DataSourceIterator*>(this);

    // Fast path: Check if we have valid batch data
    const bool HAS_VALID_BATCH = check_batch_validity(iter);
    if (HAS_VALID_BATCH || !iter->table_handle) {
        return HAS_VALID_BATCH;
    }

    // Need to fetch new batch
    const TupleDesc TUPLE_DESC = get_table_handle_tupledesc(iter->table_handle);

    prepare_new_batch(iter, TUPLE_DESC);
    read_and_fill_batch(iter, TUPLE_DESC);
    return finalize_batch(iter);
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

    const size_t ROW_IDX = iter->current_row_in_batch;
    const size_t NUM_COLUMNS = iter->columns.size();

    PGX_LOG(RUNTIME, DEBUG, "Accessing row %zu/%zu from batch (batch has %zu JSON columns)", ROW_IDX,
            iter->batch->num_rows, NUM_COLUMNS);

    // RecordBatchInfo structure (from lingodb):
    // [numRows: size_t][columnInfo[0]...][columnInfo[1]...]...
    // Each columnInfo has 5 fields: offset, validMultiplier, validBuffer, dataBuffer, varLenBuffer
    auto *const ROW_DATA_PTR = reinterpret_cast<size_t*>(row_data);
    ROW_DATA_PTR[0] = 1;

    for (size_t col = 0; col < NUM_COLUMNS; ++col) {
        constexpr size_t COLUMN_OFFSET_IDX = 0;
        constexpr size_t VALID_MULTIPLIER_IDX = 1;
        constexpr size_t VALID_BUFFER_IDX = 2;
        constexpr size_t DATA_BUFFER_IDX = 3;
        constexpr size_t VARLEN_BUFFER_IDX = 4;
        constexpr size_t COLUMN_INFO_SIZE = 5;

        size_t* const column_info_ptr = &ROW_DATA_PTR[1 + col * COLUMN_INFO_SIZE];

        column_info_ptr[COLUMN_OFFSET_IDX] = 0;
        column_info_ptr[VALID_MULTIPLIER_IDX] = 0; // validMultiplier (unused)

        column_info_ptr[VALID_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->column_nulls[col][ROW_IDX]);

        // For string columns, pass Arrow format pointers (length and data pointer)
        // For simple types, Datum contains the value directly
        if (iter->columns[col].type == ::ColumnType::STRING) {
            // Arrow format: DATA_BUFFER points to int32_t length, VARLEN_BUFFER points to uint8_t* data pointer
            column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->string_lengths[col][ROW_IDX]);
            column_info_ptr[VARLEN_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->string_data_ptrs[col][ROW_IDX]);

            PGX_LOG(RUNTIME, TRACE, "access() col=%zu STRING (Arrow format):", col);
            PGX_LOG(RUNTIME, TRACE, "  Length: %d (at %p)", iter->batch->string_lengths[col][ROW_IDX],
                    &iter->batch->string_lengths[col][ROW_IDX]);
            PGX_LOG(RUNTIME, TRACE, "  Data pointer: %p (at %p)", iter->batch->string_data_ptrs[col][ROW_IDX],
                    &iter->batch->string_data_ptrs[col][ROW_IDX]);
            PGX_LOG(RUNTIME, TRACE, "  DATA_BUFFER_IDX → %p", reinterpret_cast<void*>(column_info_ptr[DATA_BUFFER_IDX]));
            PGX_LOG(RUNTIME, TRACE, "  VARLEN_BUFFER_IDX → %p",
                    reinterpret_cast<void*>(column_info_ptr[VARLEN_BUFFER_IDX]));
        } else if (iter->columns[col].type == ::ColumnType::DECIMAL) {
            column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->decimal_values[col][ROW_IDX]);
            column_info_ptr[VARLEN_BUFFER_IDX] = 0;

            PGX_LOG(RUNTIME, DEBUG, "access() col=%zu DECIMAL i128 at %p, value=%lld", col,
                    &iter->batch->decimal_values[col][ROW_IDX],
                    static_cast<long long>(iter->batch->decimal_values[col][ROW_IDX]));
        } else {
            // Pass address of Datum itself (contains value)
            column_info_ptr[DATA_BUFFER_IDX] = reinterpret_cast<size_t>(&iter->batch->column_values[col][ROW_IDX]);
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
void* get_execution_context() {
    PGX_IO(RUNTIME);
    return rt_get_execution_context();
}

void set_execution_context(void* context) {
    PGX_IO(RUNTIME);
    rt_set_execution_context(context);
}

} // namespace runtime
