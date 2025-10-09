#ifndef PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H
#define PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H

#include "lingodb/runtime/DataSourceIteration.h"
#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/helpers.h"
#include <string>
#include <functional>

namespace pgx_lower::compiler::runtime {

class PostgreSQLDataSource : public ::runtime::DataSource {
    std::string tableName;
    void* scanContext;

   public:
    PostgreSQLDataSource(const std::string& description);

    void* getNext() override; // was: std::shared_ptr<arrow::RecordBatch> - Arrow stubbed out

    ~PostgreSQLDataSource() override;

    static DataSource* createFromDescription(::runtime::VarLen32 description);
};

} // namespace pgx_lower::compiler::runtime

namespace runtime {
inline DataSource* DataSource::get(VarLen32 description) {
    return pgx_lower::compiler::runtime::PostgreSQLDataSource::createFromDescription(description);
}
} // namespace runtime

#endif // PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H