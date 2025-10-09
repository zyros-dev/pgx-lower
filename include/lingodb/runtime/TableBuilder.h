#ifndef RUNTIME_TABLEBUILDER_H
#define RUNTIME_TABLEBUILDER_H
#include "lingodb/runtime/helpers.h"

#include <cassert>

namespace runtime {
class TableBuilder {
   static constexpr size_t maxBatchSize = 100000;
   void* schema; // was: std::shared_ptr<arrow::Schema> schema;
   void* batches; // was: std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   void* batchBuilder; // was: std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t currentBatchSize = 0;
   size_t currColumn = 0;

   TableBuilder(void* schema) : schema(schema) {
   }
   
   void* convertBatch(void* recordBatch) {
      return nullptr;
   }
   
   void flushBatch() {
      currentBatchSize = 0;
   }
   
   template <typename T>
   T* getBuilder() {
      return nullptr;
   }
   
   void handleStatus(int status) {
   }

   public:
   static TableBuilder* create(VarLen32 schemaDescription);
   static void destroy(TableBuilder* tb);
   void* build();

   void addBool(bool is_valid, bool value);
   void addInt8(bool is_valid, int8_t);
   void addInt16(bool is_valid, int16_t);
   void addInt32(bool is_valid, int32_t);
   void addInt64(bool is_valid, int64_t);
   void addFloat32(bool is_valid, float);
   void addFloat64(bool is_valid, double);
   void addDecimal(bool is_valid, __int128);
   void addFixedSized(bool is_valid, int64_t);
   void addBinary(bool is_valid, runtime::VarLen32);
   void setNextDecimalScale(int32_t scale);
   void nextRow();
};
} // end namespace runtime
#endif //RUNTIME_TABLEBUILDER_H