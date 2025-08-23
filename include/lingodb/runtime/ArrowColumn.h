
#ifndef PGX_LOWER_RUNTIME_POSTGRESQL_COLUMN_H
#define PGX_LOWER_RUNTIME_POSTGRESQL_COLUMN_H
#include "helpers.h"
#include "PostgreSQLBatch.h"
#include <memory>
#include <vector>

namespace pgx_lower::compiler::runtime {

class PostgreSQLColumn {
   std::shared_ptr<runtime::postgresql::PostgreSQLColumn> column;

   public:
   PostgreSQLColumn(std::shared_ptr<runtime::postgresql::PostgreSQLColumn> col) : column(col) {}
   std::shared_ptr<runtime::postgresql::PostgreSQLColumn> getColumn() const { return column; }
   
   // PostgreSQL SPI integration methods
   void* getPostgreSQLData() const { return column ? column->getPostgreSQLDatum(0) : nullptr; }
};

class PostgreSQLColumnBuilder {
   size_t numValues = 0;
   std::shared_ptr<runtime::postgresql::PostgreSQLColumn> column;
   PostgreSQLColumnBuilder* childBuilder;
   std::string pgType; // PostgreSQL type name
   std::vector<void*> additionalArrays;
   PostgreSQLColumnBuilder(const std::string& type) : pgType(type), childBuilder(nullptr) {
       column = std::make_shared<runtime::postgresql::PostgreSQLColumn>("column", type);
   }
   inline void next() { numValues++; }

   public:
   static PostgreSQLColumnBuilder* create(VarLen32 type);
   PostgreSQLColumnBuilder* getChildBuilder();
   void addBool(bool isValid, bool value);
   void addFixedSized(bool isValid, uint8_t* value);

   void addList(bool isValid);
   void addBinary(bool isValid, runtime::VarLen32);
   void merge(PostgreSQLColumnBuilder* other);
   PostgreSQLColumn* finish();
   ~PostgreSQLColumnBuilder();
   
   // PostgreSQL-specific methods
   void addInt32(bool isValid, int32_t value);
   void addInt64(bool isValid, int64_t value);
   void addFloat64(bool isValid, double value);
   void addString(bool isValid, const std::string& value);
   
   std::shared_ptr<runtime::postgresql::PostgreSQLColumn> getBuiltColumn() const { return column; }
};

} // namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_POSTGRESQL_COLUMN_H
