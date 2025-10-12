// ReSharper disable CppUseStructuredBinding
#include "lingodb/runtime/PgSortRuntime.h"
#include "pgx-lower/utility/logging.h"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>


extern "C" {
#include "postgres.h"
#include "miscadmin.h"
#include "access/tupdesc.h"
#include "access/htup_details.h"
#include "utils/tuplesort.h"
#include "executor/tuptable.h"
#include "utils/memutils.h"
#include "catalog/pg_type_d.h"
#include "varatt.h"
#include "utils/builtins.h"
#include "utils/numeric.h"
#include "fmgr.h"
}

namespace runtime {

void PgSortState::log_specification(const char* context) const {
    if (!spec) {
        PGX_LOG(RUNTIME, DEBUG, "%s: SortSpecification is NULL", context);
        return;
    }

    PGX_LOG(RUNTIME, DEBUG, "%s: SortSpecification at %p", context, spec);
    PGX_LOG(RUNTIME, DEBUG, "  num_columns=%d, num_sort_keys=%d, tuple_size=%zu", spec->num_columns,
            spec->num_sort_keys, tupleSize);

    for (int32_t i = 0; i < spec->num_columns; i++) {
        const auto& col = spec->columns[i];
        PGX_LOG(RUNTIME, DEBUG, "  Column[%d]: table='%s', column='%s', type_oid=%u, typmod=%d", i,
                col.table_name ? col.table_name : "(null)", col.column_name ? col.column_name : "(null)", col.type_oid,
                col.typmod);
    }

    for (int32_t i = 0; i < spec->num_sort_keys; i++) {
        PGX_LOG(RUNTIME, DEBUG, "  SortKey[%d]: column_idx=%d, operator_oid=%u, collation=%u, nulls_first=%s", i,
                spec->sort_key_indices[i], spec->sort_operators[i], spec->collations[i],
                spec->nulls_first[i] ? "true" : "false");
    }


    if (!column_layouts_.empty()) {
        PGX_LOG(RUNTIME, DEBUG, "  Computed Column Layouts:");
        for (size_t i = 0; i < column_layouts_.size(); i++) {
            const auto& layout = column_layouts_[i];
            auto phys_type_name = "UNKNOWN";
            switch (layout.phys_type) {
            case PhysicalType::BOOL: phys_type_name = "BOOL"; break;
            case PhysicalType::INT16: phys_type_name = "INT16"; break;
            case PhysicalType::INT32: phys_type_name = "INT32"; break;
            case PhysicalType::INT64: phys_type_name = "INT64"; break;
            case PhysicalType::FLOAT32: phys_type_name = "FLOAT32"; break;
            case PhysicalType::FLOAT64: phys_type_name = "FLOAT64"; break;
            case PhysicalType::VARLEN32: phys_type_name = "VARLEN32"; break;
            case PhysicalType::DECIMAL128: phys_type_name = "DECIMAL128"; break;
            }
            PGX_LOG(RUNTIME, DEBUG,
                    "    Layout[%zu]: tuple_offset=%zu, null_flag_offset=%zu, value_offset=%zu, "
                    "value_size=%zu, phys_type=%s, nullable=%s",
                    i, layout.tuple_offset, layout.null_flag_offset, layout.value_offset, layout.value_size,
                    phys_type_name, layout.is_nullable ? "true" : "false");
        }
    }
}

void PgSortState::compute_column_layouts() {
    PGX_IO(RUNTIME);

    column_layouts_.clear();
    column_layouts_.reserve(spec->num_columns);

    size_t current_offset = 0;

    for (int32_t i = 0; i < spec->num_columns; i++) {
        ColumnLayout layout{};
        layout.pg_type_oid = spec->columns[i].type_oid;
        layout.phys_type = get_physical_type(layout.pg_type_oid);
        layout.value_size = get_physical_size(layout.pg_type_oid);
        layout.is_nullable = true;


        layout.tuple_offset = current_offset;
        layout.null_flag_offset = current_offset;
        current_offset += 1;

        layout.value_offset = current_offset;
        current_offset += layout.value_size;

        column_layouts_.push_back(layout);

        PGX_LOG(RUNTIME, DEBUG,
                "compute_column_layouts: Column[%d] type_oid=%u phys_type=%d tuple_offset=%zu "
                "null_flag_offset=%zu value_offset=%zu value_size=%zu",
                i, layout.pg_type_oid, static_cast<int>(layout.phys_type), layout.tuple_offset, layout.null_flag_offset,
                layout.value_offset, layout.value_size);
    }

    PGX_LOG(RUNTIME, DEBUG, "compute_column_layouts: Total tuple size=%zu bytes (spec says %zu)", current_offset,
            tupleSize);

    if (current_offset != tupleSize) {
        PGX_LOG(RUNTIME, DEBUG, "compute_column_layouts: Correcting tuple size from %zu to %zu (added %d null flags)",
                tupleSize, current_offset, spec->num_columns);
        tupleSize = current_offset;
    }
}

void PgSortState::build_tuple_desc() {
    PGX_IO(RUNTIME);
    sortcontext = AllocSetContextCreateInternal(CurTransactionContext, "PgSortContext", ALLOCSET_DEFAULT_SIZES);

    const MemoryContext oldcontext = MemoryContextSwitchTo(static_cast<MemoryContext>(sortcontext));
    const TupleDesc td = CreateTemplateTupleDesc(spec->num_columns);

    for (int32_t i = 0; i < spec->num_columns; i++) {
        TupleDescInitEntry(td,
                           static_cast<AttrNumber>(i + 1), // 1-indexed!
                           spec->columns[i].column_name, spec->columns[i].type_oid, spec->columns[i].typmod, 0);

        PGX_LOG(RUNTIME, DEBUG, "build_tuple_desc: TupleDesc[%d]: name=%s, type_oid=%u, typmod=%d", i,
                spec->columns[i].column_name, spec->columns[i].type_oid, spec->columns[i].typmod);
    }

    tupdesc = td;

    MemoryContextSwitchTo(oldcontext);

    PGX_LOG(RUNTIME, DEBUG, "build_tuple_desc: Created TupleDesc with %d columns", spec->num_columns);
}

void PgSortState::init_tuplesort() {
    PGX_IO(RUNTIME);

    const MemoryContext oldcontext = MemoryContextSwitchTo(static_cast<MemoryContext>(sortcontext));
    auto* attNums = static_cast<AttrNumber*>(palloc(spec->num_sort_keys * sizeof(AttrNumber)));
    auto* sortOperators = static_cast<Oid*>(palloc(spec->num_sort_keys * sizeof(Oid)));
    auto* sortCollations = static_cast<Oid*>(palloc(spec->num_sort_keys * sizeof(Oid)));
    auto* nullsFirstFlags = static_cast<bool*>(palloc(spec->num_sort_keys * sizeof(bool)));

    for (int32_t i = 0; i < spec->num_sort_keys; i++) {
        attNums[i] = static_cast<AttrNumber>(spec->sort_key_indices[i] + 1);
        sortOperators[i] = spec->sort_operators[i];
        sortCollations[i] = spec->collations[i];
        nullsFirstFlags[i] = spec->nulls_first[i];

        PGX_LOG(RUNTIME, DEBUG, "init_tuplesort: SortKey[%d]: attNum=%d, op=%u, collation=%u, nulls_first=%s", i,
                attNums[i], sortOperators[i], sortCollations[i], nullsFirstFlags[i] ? "true" : "false");
    }

    Tuplesortstate* ts = tuplesort_begin_heap(static_cast<TupleDesc>(tupdesc), spec->num_sort_keys, attNums,
                                              sortOperators, sortCollations, nullsFirstFlags, work_mem, nullptr,
                                              TUPLESORT_NONE);

    sortstate = ts;

    input_slot = MakeSingleTupleTableSlot(static_cast<TupleDesc>(tupdesc), &TTSOpsHeapTuple);
    output_slot = MakeSingleTupleTableSlot(static_cast<TupleDesc>(tupdesc), &TTSOpsMinimalTuple);

    MemoryContextSwitchTo(oldcontext);

    PGX_LOG(RUNTIME, DEBUG, "init_tuplesort: Initialized with %d sort keys, work_mem=4096 KB", spec->num_sort_keys);
}

void PgSortState::unpack_mlir_to_datums(const uint8_t* mlir_tuple, void* values_ptr, bool* isnull) const {
    auto* values = static_cast<Datum*>(values_ptr);


    for (size_t i = 0; i < column_layouts_.size(); i++) {
        const auto& layout = column_layouts_[i];

        const uint8_t null_flag = mlir_tuple[layout.null_flag_offset];
        isnull[i] = (null_flag != 0);

        if (isnull[i]) {
            values[i] = static_cast<Datum>(0);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu]: NULL", i);
            continue;
        }

        switch (layout.phys_type) {
        case PhysicalType::BOOL: {
            const uint8_t val = mlir_tuple[layout.value_offset];
            values[i] = BoolGetDatum(val != 0);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] bool: value=%d", i, val);
            break;
        }
        case PhysicalType::INT16: {
            const int16_t val = *reinterpret_cast<const int16_t*>(&mlir_tuple[layout.value_offset]);
            values[i] = Int16GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] int16: value=%d", i, val);
            break;
        }
        case PhysicalType::INT32: {
            const int32_t val = *reinterpret_cast<const int32_t*>(&mlir_tuple[layout.value_offset]);
            values[i] = Int32GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] int32: value=%d", i, val);
            break;
        }
        case PhysicalType::INT64: {
            const int64_t val = *reinterpret_cast<const int64_t*>(&mlir_tuple[layout.value_offset]);
            values[i] = Int64GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] int64: value=%ld", i, val);
            break;
        }
        case PhysicalType::FLOAT32: {
            const float val = *reinterpret_cast<const float*>(&mlir_tuple[layout.value_offset]);
            values[i] = Float4GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] float32: value=%f", i, val);
            break;
        }
        case PhysicalType::FLOAT64: {
            const double val = *reinterpret_cast<const double*>(&mlir_tuple[layout.value_offset]);
            values[i] = Float8GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] float64: value=%f", i, val);
            break;
        }
        case PhysicalType::VARLEN32: {
            const uint32_t len_with_flag = *reinterpret_cast<const uint32_t*>(&mlir_tuple[layout.value_offset]);
            const size_t len = len_with_flag & ~0x80000000;
            char* str_ptr = *reinterpret_cast<char* const*>(&mlir_tuple[layout.value_offset + 8]);

            if (!str_ptr) {
                PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] string: ERROR - NULL pointer", i);
                isnull[i] = true;
                values[i] = static_cast<Datum>(0);
            } else {
                const text* pg_text = cstring_to_text_with_len(str_ptr, static_cast<int>(len));
                values[i] = PointerGetDatum(pg_text);
                PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] string: len=%zu, value='%.*s'", i, len,
                        static_cast<int>(len), str_ptr);
            }
            break;
        }
        case PhysicalType::DECIMAL128: {
            // The scale is constant per column, so ordering is preserved
            const __int128 val = *reinterpret_cast<const __int128*>(&mlir_tuple[layout.value_offset]);

            const int64_t upper = static_cast<int64_t>(val >> 64);
            const uint64_t lower = static_cast<uint64_t>(val & 0xFFFFFFFFFFFFFFFFULL);

            const int64_t lower_high = static_cast<int64_t>(lower >> 32);
            const int64_t lower_low = lower & 0xFFFFFFFFULL;

            const Numeric num_upper = int64_to_numeric(upper);
            const Numeric num_2_64 = int64_to_numeric(1LL << 63);
            Datum upper_shifted = DirectFunctionCall2(numeric_mul,
                                                      NumericGetDatum(num_upper),
                                                      NumericGetDatum(num_2_64));
            upper_shifted = DirectFunctionCall2(numeric_mul,
                                                upper_shifted,
                                                NumericGetDatum(int64_to_numeric(2)));

            const Numeric num_lower_high = int64_to_numeric(lower_high);
            const Numeric num_2_32 = int64_to_numeric(1LL << 32);
            const Datum lower_high_shifted = DirectFunctionCall2(numeric_mul,
                                                           NumericGetDatum(num_lower_high),
                                                           NumericGetDatum(num_2_32));

            // Add all parts: upper * 2^64 + lower_high * 2^32 + lower_low
            const Datum temp = DirectFunctionCall2(numeric_add, upper_shifted, lower_high_shifted);
            const Datum result = DirectFunctionCall2(numeric_add,
                                                     temp,
                                                     NumericGetDatum(int64_to_numeric(lower_low)));

            values[i] = result;
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] decimal128: upper=%ld, lower=%lu", i, upper, lower);
            break;
        }
        default:
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu]: Unknown physical type %d", i,
                    static_cast<int>(layout.phys_type));
            isnull[i] = true;
            values[i] = static_cast<Datum>(0);
            break;
        }
    }
}

void PgSortState::pack_datums_to_mlir(void* values_ptr, const bool* isnull, uint8_t* mlir_tuple) const {
    const auto* values = static_cast<Datum*>(values_ptr);

    memset(mlir_tuple, 0, tupleSize);
    for (size_t i = 0; i < column_layouts_.size(); i++) {
        const auto& layout = column_layouts_[i];
        mlir_tuple[layout.null_flag_offset] = isnull[i] ? 1 : 0;

        if (isnull[i]) {
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu]: NULL", i);
            continue;
        }

        switch (layout.phys_type) {
        case PhysicalType::BOOL: {
            const bool val = DatumGetBool(values[i]);
            mlir_tuple[layout.value_offset] = val ? 1 : 0;
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] bool: value=%d", i, val);
            break;
        }
        case PhysicalType::INT16: {
            const int16_t val = DatumGetInt16(values[i]);
            *reinterpret_cast<int16_t*>(&mlir_tuple[layout.value_offset]) = val;
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] int16: value=%d", i, val);
            break;
        }
        case PhysicalType::INT32: {
            const int32_t val = DatumGetInt32(values[i]);
            *reinterpret_cast<int32_t*>(&mlir_tuple[layout.value_offset]) = val;
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] int32: value=%d", i, val);
            break;
        }
        case PhysicalType::INT64: {
            const int64_t val = DatumGetInt64(values[i]);
            *reinterpret_cast<int64_t*>(&mlir_tuple[layout.value_offset]) = val;
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] int64: value=%ld", i, val);
            break;
        }
        case PhysicalType::FLOAT32: {
            const float val = DatumGetFloat4(values[i]);
            *reinterpret_cast<float*>(&mlir_tuple[layout.value_offset]) = val;
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] float32: value=%f", i, val);
            break;
        }
        case PhysicalType::FLOAT64: {
            const double val = DatumGetFloat8(values[i]);
            *reinterpret_cast<double*>(&mlir_tuple[layout.value_offset]) = val;
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] float64: value=%f", i, val);
            break;
        }
        case PhysicalType::VARLEN32: {
            const auto pg_text = DatumGetTextPP(values[i]);
            const int len = VARSIZE_ANY_EXHDR(pg_text);
            const char* str_data = VARDATA_ANY(pg_text);

            const auto new_str = static_cast<char*>(palloc(len + 1));
            memcpy(new_str, str_data, len);
            new_str[len] = '\0';

            const uint32_t len_with_flag = len | 0x80000000;
            *reinterpret_cast<uint32_t*>(&mlir_tuple[layout.value_offset]) = len_with_flag;
            *reinterpret_cast<char**>(&mlir_tuple[layout.value_offset + 8]) = new_str;

            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] string: len=%d, value='%.*s', ptr=%p", i, len, len, new_str,
                    new_str);
            break;
        }
        case PhysicalType::DECIMAL128: {
            const Numeric num = DatumGetNumeric(values[i]);

            const Numeric num_shift = int64_to_numeric(1LL << 63);
            const Datum divisor_datum = DirectFunctionCall2(numeric_mul,
                                                      NumericGetDatum(num_shift),
                                                      NumericGetDatum(int64_to_numeric(2)));  // 2^64

            const Datum upper_datum = DirectFunctionCall2(numeric_div_trunc,
                                                    NumericGetDatum(num),
                                                    divisor_datum);

            bool error = false;
            const int64_t upper = numeric_int8_opt_error(DatumGetNumeric(upper_datum), &error);

            const Datum lower_datum = DirectFunctionCall2(numeric_mod,
                                                    NumericGetDatum(num),
                                                    divisor_datum);
            const int64_t lower = numeric_int8_opt_error(DatumGetNumeric(lower_datum), &error);

            if (!error) {
                const __int128 val = (static_cast<__int128>(upper) << 64) | static_cast<uint64_t>(lower);
                *reinterpret_cast<__int128*>(&mlir_tuple[layout.value_offset]) = val;
                PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] decimal128: upper=%ld, lower=%ld", i, upper, lower);
            } else {
                memset(&mlir_tuple[layout.value_offset], 0, 16);
                PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu] decimal128: ERROR converting", i);
            }
            break;
        }
        default:
            PGX_LOG(RUNTIME, DEBUG, "  pack Column[%zu]: Unknown physical type %d", i,
                    static_cast<int>(layout.phys_type));
            break;
        }
    }
}

PgSortState::PgSortState(const SortSpecification* spec, size_t tupleSize, size_t initialCapacity)
: tupleSize(tupleSize)
, tupleCount(0)
, capacity(initialCapacity)
, spec(spec)
, sortcontext(nullptr)
, tupdesc(nullptr)
, sortstate(nullptr)
, input_slot(nullptr)
, output_slot(nullptr)
, sorted(false)
, fetch_index(0) {
    PGX_IO(RUNTIME);

    log_specification("PgSortState::PgSortState");
    compute_column_layouts();
    build_tuple_desc();
    init_tuplesort();

    PGX_LOG(RUNTIME, DEBUG,
            "PgSortState::PgSortState: Created with tupleSize=%zu, initialCapacity=%zu, bufferSize=%zu bytes",
            tupleSize, initialCapacity, tupleSize * initialCapacity);
}

PgSortState::~PgSortState() {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "PgSortState::~PgSortState: Destroying state with %zu tuples", tupleCount);

    if (sortcontext) {
        MemoryContextDelete(static_cast<MemoryContext>(sortcontext));
        sortcontext = nullptr;
        tupdesc = nullptr;
        sortstate = nullptr;
        input_slot = nullptr;
        output_slot = nullptr;
    }
}

void PgSortState::appendTuple(const uint8_t* tupleData) {
    PGX_IO(RUNTIME);

    if (!tupleData) {
        PGX_LOG(RUNTIME, DEBUG, "appendTuple: NULL tuple data, skipping");
        return;
    }

    if (tupleCount == 0) {
        log_specification("PgSortState::appendTuple (first tuple)");
    }

    std::ostringstream hexdump;
    const size_t dumpSize = std::min(tupleSize, static_cast<size_t>(32));
    for (size_t i = 0; i < dumpSize; ++i) {
        hexdump << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(tupleData[i]) << " ";
    }
    PGX_LOG(RUNTIME, DEBUG, "appendTuple: tuple #%zu, hexdump: %s%s", tupleCount, hexdump.str().c_str(),
            (tupleSize > 32 ? "..." : ""));
    // Pre-allocate saved string data for all VarLen32 columns
    struct SavedString {
        char* data;
        size_t len;
    };
    std::vector<SavedString> saved_strings(spec->num_columns, {nullptr, 0});

    for (size_t i = 0; i < column_layouts_.size(); i++) {
        const auto& layout = column_layouts_[i];
        if (layout.phys_type == PhysicalType::VARLEN32) {
            const uint8_t null_flag = tupleData[layout.null_flag_offset];
            if (null_flag == 0) { // Not null
                // VarLen32 layout: len at offset+0, pointer at offset+8
                const uint32_t len_with_flag = *reinterpret_cast<const uint32_t*>(&tupleData[layout.value_offset]);
                const size_t len = len_with_flag & ~0x80000000;

                if (char* str_ptr = *reinterpret_cast<char* const*>(&tupleData[layout.value_offset + 8])) {
                    saved_strings[i].len = len;
                    saved_strings[i].data = static_cast<char*>(malloc(len + 1));
                    if (saved_strings[i].data) {
                        memcpy(saved_strings[i].data, str_ptr, len);
                        saved_strings[i].data[len] = '\0';
                        PGX_LOG(RUNTIME, DEBUG, "appendTuple: Pre-extracted Column[%zu] string: len=%zu, value='%.*s'",
                                i, len, static_cast<int>(len), saved_strings[i].data);
                    }
                }
            }
        }
    }

    const MemoryContext oldcontext = MemoryContextSwitchTo(static_cast<MemoryContext>(sortcontext));
    auto* values = static_cast<Datum*>(palloc(spec->num_columns * sizeof(Datum)));
    auto* isnull = static_cast<bool*>(palloc(spec->num_columns * sizeof(bool)));
    for (size_t i = 0; i < column_layouts_.size(); i++) {
        const auto& layout = column_layouts_[i];

        const uint8_t null_flag = tupleData[layout.null_flag_offset];
        isnull[i] = (null_flag != 0);

        if (isnull[i]) {
            values[i] = static_cast<Datum>(0);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu]: NULL", i);
            continue;
        }

        switch (layout.phys_type) {
        case PhysicalType::BOOL: {
            const uint8_t val = tupleData[layout.value_offset];
            values[i] = BoolGetDatum(val != 0);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] bool: value=%d", i, val);
            break;
        }
        case PhysicalType::INT16: {
            const int16_t val = *reinterpret_cast<const int16_t*>(&tupleData[layout.value_offset]);
            values[i] = Int16GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] int16: value=%d", i, val);
            break;
        }
        case PhysicalType::INT32: {
            const int32_t val = *reinterpret_cast<const int32_t*>(&tupleData[layout.value_offset]);
            values[i] = Int32GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] int32: value=%d", i, val);
            break;
        }
        case PhysicalType::INT64: {
            const int64_t val = *reinterpret_cast<const int64_t*>(&tupleData[layout.value_offset]);
            values[i] = Int64GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] int64: value=%ld", i, val);
            break;
        }
        case PhysicalType::FLOAT32: {
            const float val = *reinterpret_cast<const float*>(&tupleData[layout.value_offset]);
            values[i] = Float4GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] float32: value=%f", i, val);
            break;
        }
        case PhysicalType::FLOAT64: {
            const double val = *reinterpret_cast<const double*>(&tupleData[layout.value_offset]);
            values[i] = Float8GetDatum(val);
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] float64: value=%f", i, val);
            break;
        }
        case PhysicalType::VARLEN32: {
            if (saved_strings[i].data) {
                const text* pg_text = cstring_to_text_with_len(saved_strings[i].data,
                                                               static_cast<int>(saved_strings[i].len));
                values[i] = PointerGetDatum(pg_text);
                PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] string: len=%zu, value='%s'", i, saved_strings[i].len,
                        saved_strings[i].data);
            } else {
                values[i] = static_cast<Datum>(0);
                isnull[i] = true;
                PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] string: NULL pointer", i);
            }
            break;
        }
        case PhysicalType::DECIMAL128: {
            // Read i128 from MLIR tuple and convert to Numeric
            const __int128 val = *reinterpret_cast<const __int128*>(&tupleData[layout.value_offset]);

            const int64_t upper = static_cast<int64_t>(val >> 64);
            const uint64_t lower = static_cast<uint64_t>(val & 0xFFFFFFFFFFFFFFFFULL);

            // Build result: upper * 2^64 + lower
            const int64_t lower_high = static_cast<int64_t>(lower >> 32);
            const int64_t lower_low = lower & 0xFFFFFFFFULL;

            // upper * 2^64
            const Numeric num_upper = int64_to_numeric(upper);
            const Numeric num_2_64 = int64_to_numeric(1LL << 63);
            Datum upper_shifted = DirectFunctionCall2(numeric_mul,
                                                      NumericGetDatum(num_upper),
                                                      NumericGetDatum(num_2_64));
            upper_shifted = DirectFunctionCall2(numeric_mul,
                                                upper_shifted,
                                                NumericGetDatum(int64_to_numeric(2)));

            // lower_high * 2^32
            const Numeric num_lower_high = int64_to_numeric(lower_high);
            const Numeric num_2_32 = int64_to_numeric(1LL << 32);
            Datum lower_high_shifted = DirectFunctionCall2(numeric_mul,
                                                           NumericGetDatum(num_lower_high),
                                                           NumericGetDatum(num_2_32));

            // Add all parts
            Datum temp = DirectFunctionCall2(numeric_add, upper_shifted, lower_high_shifted);
            const Datum result = DirectFunctionCall2(numeric_add,
                                                     temp,
                                                     NumericGetDatum(int64_to_numeric(lower_low)));

            values[i] = result;
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu] decimal128: i128 val, upper=%ld, lower=%lu", i, upper, lower);
            break;
        }
        default:
            PGX_LOG(RUNTIME, DEBUG, "  unpack Column[%zu]: Unknown physical type %d", i,
                    static_cast<int>(layout.phys_type));
            isnull[i] = true;
            values[i] = static_cast<Datum>(0);
            break;
        }
    }

    for (auto& saved : saved_strings) {
        if (saved.data) {
            free(saved.data);
        }
    }

    const HeapTuple htup = heap_form_tuple(static_cast<TupleDesc>(tupdesc), values, isnull);
    ExecStoreHeapTuple(htup, static_cast<TupleTableSlot*>(input_slot), false);
    tuplesort_puttupleslot(static_cast<Tuplesortstate*>(sortstate), static_cast<TupleTableSlot*>(input_slot));
    MemoryContextSwitchTo(oldcontext);
    tupleCount++;

    PGX_LOG(RUNTIME, DEBUG, "appendTuple: Successfully appended tuple #%zu", tupleCount);
}

void PgSortState::flushTuples() const {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "PgSortState::flushTuples: Flushing %zu tuples (stub - no actual flush)", tupleCount);
}

void PgSortState::performSort() {
    PGX_IO(RUNTIME);
    if (sorted) {
        PGX_LOG(RUNTIME, DEBUG, "performSort: Already sorted, skipping");
        return;
    }

    log_specification("PgSortState::performSort");
    PGX_LOG(RUNTIME, DEBUG, "performSort: Sorting %zu tuples...", tupleCount);

    const MemoryContext oldcontext = MemoryContextSwitchTo(static_cast<MemoryContext>(sortcontext));
    tuplesort_performsort(static_cast<Tuplesortstate*>(sortstate));
    MemoryContextSwitchTo(oldcontext);

    sorted = true;
    fetch_index = 0;
    PGX_LOG(RUNTIME, DEBUG, "performSort: Sort complete!");
}

void* PgSortState::getNextTuple() {
    PGX_IO(RUNTIME);

    if (!sorted) {
        PGX_LOG(RUNTIME, DEBUG, "getNextTuple: Not sorted yet, returning nullptr");
        return nullptr;
    }

    const MemoryContext oldcontext = MemoryContextSwitchTo(static_cast<MemoryContext>(sortcontext));
    const bool got_tuple = tuplesort_gettupleslot(static_cast<Tuplesortstate*>(sortstate),
                                                  true,
                                                  false,
                                                  static_cast<TupleTableSlot*>(output_slot),
                                                  nullptr
    );

    if (!got_tuple) {
        MemoryContextSwitchTo(oldcontext);
        PGX_LOG(RUNTIME, DEBUG, "getNextTuple: No more tuples (fetched %zu total)", fetch_index);
        return nullptr;
    }

    slot_getallattrs(static_cast<TupleTableSlot*>(output_slot));
    const auto* slot = static_cast<TupleTableSlot*>(output_slot);
    Datum* values = slot->tts_values;
    bool* isnull = slot->tts_isnull;

    auto* mlir_tuple = static_cast<uint8_t*>(palloc(tupleSize));
    pack_datums_to_mlir(values, isnull, mlir_tuple);
    MemoryContextSwitchTo(oldcontext);
    fetch_index++;

    std::ostringstream hexdump;
    const size_t dumpSize = std::min(tupleSize, static_cast<size_t>(32));
    for (size_t i = 0; i < dumpSize; ++i) {
        hexdump << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mlir_tuple[i]) << " ";
    }
    PGX_LOG(RUNTIME, DEBUG, "getNextTuple: Returned sorted tuple #%zu, hexdump: %s%s", fetch_index,
            hexdump.str().c_str(), (tupleSize > 32 ? "..." : ""));

    return mlir_tuple;
}

PgSortState* PgSortState::create(size_t tupleSize, uint64_t specPtr) {
    PGX_IO(RUNTIME);
    PGX_LOG(RUNTIME, DEBUG, "PgSortState::create: tupleSize=%zu, specPtr=0x%lx", tupleSize, specPtr);

    const auto spec = reinterpret_cast<const SortSpecification*>(specPtr);

    constexpr size_t DEFAULT_CAPACITY = 1024;
    return new PgSortState(spec, tupleSize, DEFAULT_CAPACITY);
}

void PgSortState::destroy(PgSortState* state) {
    PGX_IO(RUNTIME);
    if (!state) {
        PGX_LOG(RUNTIME, DEBUG, "PgSortState::destroy: NULL state pointer");
        return;
    }

    state->log_specification("PgSortState::destroy");
    PGX_LOG(RUNTIME, DEBUG, "PgSortState::destroy: Destroying state at %p", state);
    delete state;
}

} // namespace runtime
