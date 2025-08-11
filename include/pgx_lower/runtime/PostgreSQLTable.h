#ifndef PGX_LOWER_RUNTIME_POSTGRESQLTABLE_H
#define PGX_LOWER_RUNTIME_POSTGRESQLTABLE_H
#include "ThreadLocal.h"
#include "helpers.h"
#include "runtime/ArrowColumn.h"
#include <memory>

namespace pgx_lower::compiler::runtime {
class PostgreSQLTable {
   void* postgresqlResult;  // Stub: PostgreSQL result set

   public:
   PostgreSQLTable(void* result) : postgresqlResult(result) {};
   static PostgreSQLTable* createEmpty();
   PostgreSQLTable* addColumn(VarLen32 name, ArrowColumn* column);  // TODO: Change ArrowColumn to PostgreSQLColumn
   ArrowColumn* getColumn(VarLen32 name);  // TODO: Change return type to PostgreSQLColumn
   PostgreSQLTable* merge(ThreadLocal* threadLocal);
   void* get() const { return postgresqlResult; }  // Stub implementation
};
} // end namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_POSTGRESQLTABLE_H
