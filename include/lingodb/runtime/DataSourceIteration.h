#ifndef RUNTIME_DATASOURCEITERATION_H
#define RUNTIME_DATASOURCEITERATION_H
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/helpers.h"
namespace runtime {
class DataSource {
   public:
   virtual void* getNext() = 0;
   virtual ~DataSource() {}
   
   static DataSource* get(VarLen32 description);
};
class DataSourceIteration {
   void* currChunk;
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
   static DataSourceIteration* start(ExecutionContext* executionContext, runtime::VarLen32 varlen32_param);
   bool isValid();
   void next();
   void access(RecordBatchInfo* info);
   static void end(DataSourceIteration*);
};
} // end namespace runtime
#endif // RUNTIME_DATASOURCEITERATION_H
