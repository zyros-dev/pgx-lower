#include "pgx-lower/runtime/PostgreSQLDataSource.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/utility/logging.h"
#include <cstring>
#include <sstream>


namespace pgx_lower::compiler::runtime {

PostgreSQLDataSource::PostgreSQLDataSource(const std::string& description) : scanContext(nullptr) {
    size_t table_pos = description.find(R"("table": ")");
    if (table_pos != std::string::npos) {
        table_pos += 10; // Skip past "table": "
        size_t const end_pos = description.find('\"', table_pos);
        if (end_pos != std::string::npos) {
            std::string const full_table_spec = description.substr(table_pos, end_pos - table_pos);
            
            size_t const pipe_pos = full_table_spec.find("|oid:");
            if (pipe_pos != std::string::npos) {
                tableName = full_table_spec.substr(0, pipe_pos);
                std::string const oid_str = full_table_spec.substr(pipe_pos + 5);
                Oid const table_oid = static_cast<Oid>(std::stoul(oid_str));
                ::g_jit_table_oid = table_oid;
                PGX_LOG(RUNTIME, DEBUG, "Extracted table='%s', OID=%u from spec='%s'", 
                        tableName.c_str(), table_oid, full_table_spec.c_str());
            } else {
                // No OID in spec, just use the table name
                tableName = full_table_spec;
                PGX_LOG(RUNTIME, DEBUG, "No OID in table spec, using table='%s'", tableName.c_str());
            }
        }
    }
    scanContext = open_postgres_table(tableName.c_str());
}

void* PostgreSQLDataSource::getNext() {
    return nullptr;
}

PostgreSQLDataSource::~PostgreSQLDataSource() {
    if (scanContext) {
        PGX_LOG(RUNTIME, DEBUG, "[PostgreSQLDataSource] Closing PostgreSQL table: %s", tableName.c_str());
        close_postgres_table(scanContext);
    }
    PGX_LOG(RUNTIME, DEBUG, "[PostgreSQLDataSource] PostgreSQL data source destroyed");
}

::runtime::DataSource* PostgreSQLDataSource::createFromDescription(::runtime::VarLen32 description) {
    // Convert VarLen32 to string
    std::string const desc_str(description.data(), description.getLen());
    
    PGX_LOG(RUNTIME, DEBUG, "[PostgreSQLDataSource] Creating data source from description");
    // Return PostgreSQL-specific data source
    return new PostgreSQLDataSource(desc_str);
}

} // namespace pgx_lower::compiler::runtime