#ifndef PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H
#define PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H

#include "runtime/DataSourceIteration.h"
#include "runtime/helpers.h"
#include <string>
#include <functional>

namespace pgx_lower::compiler::runtime {

// PostgreSQL implementation of DataSource
class PostgreSQLDataSource : public DataSource {
    std::string tableName;
    void* scanContext;
    
public:
    PostgreSQLDataSource(const std::string& description);
    
    void iterate(bool parallel, std::vector<std::string> members, 
                const std::function<void(BatchView*)>& cb) override;
    
    ~PostgreSQLDataSource() override;
    
    // Static factory method that DataSource::get will call
    static DataSource* createFromDescription(runtime::VarLen32 description);
};

// This will be called from GetExternalOp lowering
inline DataSource* DataSource::get(runtime::VarLen32 description) {
    return PostgreSQLDataSource::createFromDescription(description);
}

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_POSTGRESQLDATASOURCE_H