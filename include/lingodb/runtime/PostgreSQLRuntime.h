#ifndef RUNTIME_POSTGRESQLRUNTIME_H
#define RUNTIME_POSTGRESQLRUNTIME_H
#include "lingodb/runtime/helpers.h"
#include <cstdint>

namespace runtime {

// Forward declarations for integration with DataSourceIteration
class DataSourceIteration;
class ExecutionContext;
struct RecordBatchInfo;

struct TableBuilder {
   // Data members - matching the internal structure from runtime_templates.h
   void* data;
   int64_t row_count;
   int32_t current_column_index;
   int32_t total_columns;
   
   // Constructor/Destructor
   TableBuilder();
   ~TableBuilder() = default;
   
   // Table building functions
   static TableBuilder* create(VarLen32 schema);  // Static factory method
   static void destroy(void* builder);            // Static cleanup method
   TableBuilder* build();                         // Member function returning this
   void nextRow();                                // Member function
   
   // Add data functions - non-static member functions
   void addBool(bool isValid, bool value);
   void addInt8(bool isValid, int8_t value);
   void addInt16(bool isValid, int16_t value);
   void addInt32(bool isValid, int32_t value);
   void addInt64(bool isValid, int64_t value);
   void addFloat32(bool isValid, float value);
   void addFloat64(bool isValid, double value);
   void addDecimal(bool isValid, __int128 value);
   void addFixedSized(bool isValid, int64_t value);
   void addBinary(bool isValid, VarLen32 value);
};

// DataSourceIteration is defined in DataSourceIteration.h
// We just add our static method implementations in the .cpp file

// Global context functions (standalone in runtime namespace)
void setExecutionContext(void* context);
void* getExecutionContext();

} // namespace runtime
#endif // RUNTIME_POSTGRESQLRUNTIME_H