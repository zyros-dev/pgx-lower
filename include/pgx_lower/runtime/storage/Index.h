#ifndef PGX_LOWER_RUNTIME_STORAGE_INDEX_H
#define PGX_LOWER_RUNTIME_STORAGE_INDEX_H
#include <arrow/type_fwd.h>
namespace pgx_lower::compiler::runtime {
class Index {
   public:
   virtual void ensureLoaded() = 0;
   virtual void bulkInsert(size_t startRowId, std::shared_ptr<arrow::Table> newRows) = 0;
   virtual void appendRows(size_t startRowId, std::shared_ptr<arrow::RecordBatch> newRows) = 0;
   virtual ~Index() {}
};
} // namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_STORAGE_INDEX_H
