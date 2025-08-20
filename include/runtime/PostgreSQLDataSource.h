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
    
    void* getNext() override; // was: std::shared_ptr<arrow::RecordBatch> - Arrow stubbed out
    
    ~PostgreSQLDataSource() override;
    
    // Static factory method that DataSource::get will call
    static ::runtime::DataSource* createFromDescription(::runtime::VarLen32 description);
};

} // namespace pgx_lower::compiler::runtime

// This will be called from GetExternalOp lowering
// Define the get method in the correct namespace
namespace runtime {
inline DataSource* DataSource::get(VarLen32 description) {
    return ::pgx_lower::compiler::runtime::PostgreSQLDataSource::createFromDescription(description);
}
} // namespace runtime

#endif // PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H