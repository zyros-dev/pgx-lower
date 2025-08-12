#ifndef PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H
#define PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H

#include "runtime/DataSourceIteration.h"
#include "runtime/ArrowView.h"
#include "runtime/helpers.h"
#include <string>
#include <functional>

namespace pgx_lower::compiler::runtime {

// PostgreSQL implementation of DataSource
class PostgreSQLDataSource : public ::runtime::DataSource {
    std::string tableName;
    void* scanContext;
    
public:
    PostgreSQLDataSource(const std::string& description);
    
    std::shared_ptr<arrow::RecordBatch> getNext() override;
    
    ~PostgreSQLDataSource() override;
    
    // Static factory method that DataSource::get will call
    static ::runtime::DataSource* createFromDescription(::runtime::VarLen32 description);
};

// This will be called from GetExternalOp lowering
inline ::runtime::DataSource* ::runtime::DataSource::get(::runtime::VarLen32 description) {
    return PostgreSQLDataSource::createFromDescription(description);
}

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H