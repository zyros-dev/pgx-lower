#ifndef RUNTIME_POSTGRESQLRUNTIME_H
#define RUNTIME_POSTGRESQLRUNTIME_H
#include <cstdint>
#include <optional>

#include "lingodb/runtime/RuntimeSpecifications.h"
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/runtime/temporal_types.h"

namespace runtime {

// Forward declarations for integration with DataSourceIteration
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
   void addNumericDatum(bool is_valid, NumericDatumCarrier value);
   void addInterval(bool is_valid, const pgx_lower::runtime::PgIntervalValue* value);
   void addIntervalFields(bool is_valid, int64_t time, int32_t day, int32_t month);
   void addFixedSized(bool is_valid, int64_t value);
   void addBinary(bool is_valid, VarLen32 value);
   void setNextDecimalScale(int32_t scale);
};

struct PgRowRuntime {
   static void* scanStart(int32_t relid);
   static bool scanNext(void* scan);
   static void scanEnd(void* scan);

   static int32_t getInt32Value(void* scan, int32_t fieldIndex, int32_t relid, int32_t attno, int32_t oid,
                                int32_t typmod, int32_t collation, bool nullable);
   static bool getInt32IsNull(void* scan, int32_t fieldIndex, int32_t relid, int32_t attno, int32_t oid,
                              int32_t typmod, int32_t collation, bool nullable);
   static int64_t getInt64Value(void* scan, int32_t fieldIndex, int32_t relid, int32_t attno, int32_t oid,
                                int32_t typmod, int32_t collation, bool nullable);
   static bool getInt64IsNull(void* scan, int32_t fieldIndex, int32_t relid, int32_t attno, int32_t oid,
                              int32_t typmod, int32_t collation, bool nullable);

   static void emitRowStart(int32_t expectedColumns);
   static void emitBool(int32_t fieldIndex, bool isNull, bool value, int32_t oid, int32_t typmod, int32_t collation,
                        bool nullable);
   static void emitInt32(int32_t fieldIndex, bool isNull, int32_t value, int32_t oid, int32_t typmod,
                         int32_t collation, bool nullable);
   static void emitInt64(int32_t fieldIndex, bool isNull, int64_t value, int32_t oid, int32_t typmod,
                         int32_t collation, bool nullable);
   static void emitRowDone(int32_t expectedColumns);
};

void setExecutionContext(void* context);
void* getExecutionContext();

} // namespace runtime

extern "C" bool pgx_lower_row_first_slice_runtime_tupledesc_value_null_for_testing();
extern "C" bool pgx_lower_row_first_slice_runtime_tupledesc_mismatch_for_testing();

#endif // RUNTIME_POSTGRESQLRUNTIME_H
