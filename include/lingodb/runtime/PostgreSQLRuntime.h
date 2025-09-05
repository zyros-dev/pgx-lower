#ifndef RUNTIME_POSTGRESQLRUNTIME_H
#define RUNTIME_POSTGRESQLRUNTIME_H
#include "lingodb/runtime/helpers.h"

namespace runtime {
struct TableBuilder {
   // Table building functions
   static void* create(VarLen32 schema);  // Static factory method
   static void destroy(void* builder);     // Static cleanup method
   void* build();                          // Member function
   void nextRow();                         // Member function
   
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

// DataSourceIteration is already defined in DataSourceIteration.h
// We just need to implement its static methods in our .cpp file

// Global context functions (not in a struct as they're standalone)
void setExecutionContext(void* context);
void* getExecutionContext();

} // namespace runtime
#endif // RUNTIME_POSTGRESQLRUNTIME_H