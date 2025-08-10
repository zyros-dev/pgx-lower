
#ifndef PGX_LOWER_RUNTIME_ARROWCOLUMN_H
#define PGX_LOWER_RUNTIME_ARROWCOLUMN_H
#include "helpers.h"
#include <memory>
#include <vector>

namespace pgx_lower::compiler::runtime {

// Stub implementation - replace with PostgreSQL column handling
class ArrowColumn {
   void* data; // Placeholder for actual column data

   public:
   ArrowColumn(void* data) : data(data) {}
   void* getColumn() const { return data; }
};

class ArrowColumnBuilder {
   size_t numValues = 0;
   void* builder; // Placeholder
   ArrowColumnBuilder* childBuilder;
   void* type; // Placeholder for type info
   std::vector<void*> additionalArrays;
   ArrowColumnBuilder(void* type) : type(type), builder(nullptr), childBuilder(nullptr) {}
   inline void next() {}

   public:
   static ArrowColumnBuilder* create(VarLen32 type);
   ArrowColumnBuilder* getChildBuilder();
   void addBool(bool isValid, bool value);
   void addFixedSized(bool isValid, uint8_t* value);

   void addList(bool isValid);
   void addBinary(bool isValid, runtime::VarLen32);
   void merge(ArrowColumnBuilder* other);
   ArrowColumn* finish();
   ~ArrowColumnBuilder();
};

} // namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_ARROWCOLUMN_H
