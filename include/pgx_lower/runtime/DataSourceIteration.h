#ifndef RUNTIME_DATASOURCEITERATION_H
#define RUNTIME_DATASOURCEITERATION_H
#include "runtime/ExecutionContext.h"
#include "runtime/helpers.h"
namespace runtime {
class DataSource {
   public:
   virtual void* getNext() = 0; // was: std::shared_ptr<arrow::RecordBatch> getNext() - Arrow stubbed out
   virtual ~DataSource() {}
   
   // Static factory method - will be implemented by concrete DataSource types  
   static DataSource* get(VarLen32 description);
};
class DataSourceIteration {
   void* currChunk; // was: std::shared_ptr<arrow::RecordBatch> currChunk - Arrow stubbed out
   std::shared_ptr<DataSource> dataSource;
   std::vector<size_t> colIds;

   public:
   DataSourceIteration(const std::shared_ptr<DataSource>& dataSource, const std::vector<size_t>& colIds);

   struct ColumnInfo {
      size_t offset;
      size_t validMultiplier;
      uint8_t* validBuffer;
      uint8_t* dataBuffer;
      uint8_t* varLenBuffer;
   };
   struct RecordBatchInfo {
      size_t numRows;
      ColumnInfo columnInfo[];
   };
   static DataSourceIteration* start(ExecutionContext* executionContext, runtime::VarLen32 description);
   bool isValid();
   void next();
   void access(RecordBatchInfo* info);
   static void end(DataSourceIteration*);
};
} // end namespace runtime
#endif // RUNTIME_DATASOURCEITERATION_H
