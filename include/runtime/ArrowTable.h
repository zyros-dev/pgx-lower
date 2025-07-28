#ifndef PGX_LOWER_RUNTIME_ARROWTABLE_H
#define PGX_LOWER_RUNTIME_ARROWTABLE_H
#include "ThreadLocal.h"
#include "helpers.h"
#include "runtime/ArrowColumn.h"
#include <arrow/type_fwd.h>
namespace pgx_lower::compiler::runtime {
class ArrowTable {
   std::shared_ptr<arrow::Table> table;

   public:
   ArrowTable(std::shared_ptr<arrow::Table> table) : table(table) {};
   static ArrowTable* createEmpty();
   ArrowTable* addColumn(VarLen32 name, ArrowColumn* column);
   ArrowColumn* getColumn(VarLen32 name);
   ArrowTable* merge(ThreadLocal* threadLocal);
   std::shared_ptr<arrow::Table> get() const { return table; }
};
} // end namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_ARROWTABLE_H
