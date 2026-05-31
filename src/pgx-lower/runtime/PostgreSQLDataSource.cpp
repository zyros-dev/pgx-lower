#include "pgx-lower/runtime/PostgreSQLDataSource.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/utility/logging.h"
#include <cstring>
#include <sstream>


namespace pgx_lower::compiler::runtime {

PostgreSQLDataSource::PostgreSQLDataSource(const std::string& description) : scanContext(nullptr) {
    size_t tablePos = description.find("\"table\": \"");
    if (tablePos != std::string::npos) {
        tablePos += 10; // Skip past "table": "
        size_t endPos = description.find("\"", tablePos);
        if (endPos != std::string::npos) {
            std::string fullTableSpec = description.substr(tablePos, endPos - tablePos);
            
            size_t pipePos = fullTableSpec.find("|oid:");
            if (pipePos != std::string::npos) {
                tableName = fullTableSpec.substr(0, pipePos);
                std::string oidStr = fullTableSpec.substr(pipePos + 5);
                Oid tableOid = static_cast<Oid>(std::stoul(oidStr));
                ::g_jit_table_oid = tableOid;
                PGX_LOG(RUNTIME, DEBUG, "Extracted table='%s', OID=%u from spec='%s'", 
                        tableName.c_str(), tableOid, fullTableSpec.c_str());
            } else {
                // No OID in spec, just use the table name
                tableName = fullTableSpec;
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
    std::string descStr(description.data(), description.getLen());
    
    PGX_LOG(RUNTIME, DEBUG, "[PostgreSQLDataSource] Creating data source from description");
    // Return PostgreSQL-specific data source
    return new PostgreSQLDataSource(descStr);
}

} // namespace pgx_lower::compiler::runtime