#ifndef PGX_LOWER_RUNTIME_DATASOURCEITERATION_H
#define PGX_LOWER_RUNTIME_DATASOURCEITERATION_H
#include "ArrowTable.h"
#include "ArrowView.h"
#include "runtime/ExecutionContext.h"
#include "runtime/helpers.h"
namespace pgx_lower::compiler::runtime {
class DataSource {
   public:
   virtual void iterate(bool parallel, std::vector<std::string> members, const std::function<void(BatchView*)>& cb) = 0;
   virtual ~DataSource() {}
   static DataSource* get(runtime::VarLen32 description);
   //static DataSource* getFromTable(ArrowTable* arrowTable, runtime::VarLen32 mappingVal,runtime::VarLen32 columnArray);
};
class DataSourceIteration {
   std::shared_ptr<arrow::RecordBatch> currChunk;
   DataSource* dataSource;
   std::vector<std::string> members;

   public:
   DataSourceIteration(DataSource* dataSource, const std::vector<std::string>& members);

   static DataSourceIteration* init(DataSource* dataSource, runtime::VarLen32 members);
   static void end(DataSourceIteration*);
   void iterate(bool parallel, void (*forEachChunk)(BatchView*, void*), void*);
};
} // end namespace pgx_lower::compiler::runtime
#endif // PGX_LOWER_RUNTIME_DATASOURCEITERATION_H
