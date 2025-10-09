#ifndef RUNTIME_POSTGRESQLRUNTIME_H
#define RUNTIME_POSTGRESQLRUNTIME_H
#include "lingodb/runtime/helpers.h"
#include <cstdint>
#include <optional>

namespace runtime {

class DataSourceIteration;
class ExecutionContext;
struct RecordBatchInfo;

struct TableBuilder {
    void* data;
    int64_t row_count;
    int32_t current_column_index;
    int32_t total_columns;
    std::optional<int32_t> next_decimal_scale;

    TableBuilder();
    ~TableBuilder() = default;

    static TableBuilder* create(VarLen32 schema_param);
    static void destroy(void* builder);
    TableBuilder* build();
    void nextRow();

    void addBool(bool is_valid, bool value);
    void addInt8(bool is_valid, int8_t value);
    void addInt16(bool is_valid, int16_t value);
    void addInt32(bool is_valid, int32_t value);
    void addInt64(bool is_valid, int64_t value);
    void addFloat32(bool is_valid, float value);
    void addFloat64(bool is_valid, double value);
    void addDecimal(bool is_valid, __int128 value);
    void addFixedSized(bool is_valid, int64_t value);
    void addBinary(bool is_valid, VarLen32 value);
    void setNextDecimalScale(int32_t scale);
};

void setExecutionContext(void* context);
void* getExecutionContext();

} // namespace runtime
#endif // RUNTIME_POSTGRESQLRUNTIME_H