#ifndef RUNTIME_TABLEBUILDER_H
#define RUNTIME_TABLEBUILDER_H
#include "runtime/helpers.h"

#include <cassert>

// Arrow includes stubbed out - Arrow dependency removed
// #include <arrow/table.h>
// #include <arrow/table_builder.h>

namespace runtime {
class TableBuilder {
   static constexpr size_t maxBatchSize = 100000;
   // Arrow types stubbed out - to be replaced with pgx-lower native types
   void* schema; // was: std::shared_ptr<arrow::Schema> schema;
   void* batches; // was: std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   void* batchBuilder; // was: std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t currentBatchSize = 0;
   size_t currColumn = 0;

   // All Arrow functions stubbed out
   TableBuilder(void* schema) : schema(schema) {
      // arrow::RecordBatchBuilder::Make stubbed out
   }
   
   void* convertBatch(void* recordBatch) {
      // Arrow RecordBatch conversion stubbed out
      return nullptr;
   }
   
   void flushBatch() {
      // Arrow batch flushing stubbed out
      currentBatchSize = 0;
   }
   
   template <typename T>
   T* getBuilder() {
      // Arrow builder access stubbed out
      return nullptr;
   }
   
   void handleStatus(int status) {
      // Arrow status handling stubbed out
   }

   public:
   static TableBuilder* create(VarLen32 schemaDescription);
   static void destroy(TableBuilder* tb);
   void* build(); // was: std::shared_ptr<arrow::Table>* build();

   void addBool(bool isValid, bool value);
   void addInt8(bool isValid, int8_t);
   void addInt16(bool isValid, int16_t);
   void addInt32(bool isValid, int32_t);
   void addInt64(bool isValid, int64_t);
   void addFloat32(bool isValid, float);
   void addFloat64(bool isValid, double);
   void addDecimal(bool isValid, __int128);
   void addFixedSized(bool isValid, int64_t);
   void addBinary(bool isValid, runtime::VarLen32);
   void nextRow();
};
} // end namespace runtime
#endif //RUNTIME_TABLEBUILDER_H