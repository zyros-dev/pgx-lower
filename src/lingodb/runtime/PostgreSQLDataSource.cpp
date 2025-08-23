#include "lingodb/runtime/PostgreSQLDataSource.h"
#include "lingodb/runtime/tuple_access.h"
#include "pgx-lower/execution/logging.h"
#include <cstring>
#include <sstream>


namespace pgx_lower::compiler::runtime {

PostgreSQLDataSource::PostgreSQLDataSource(const std::string& description) : scanContext(nullptr) {
    size_t tablePos = description.find("\"table\": \"");
    if (tablePos != std::string::npos) {
        tablePos += 10; // Skip past "table": "
        size_t endPos = description.find("\"", tablePos);
        if (endPos != std::string::npos) {
            tableName = description.substr(tablePos, endPos - tablePos);
        }
    }
    size_t oidPos = description.find("\"oid\": \"");
    if (oidPos != std::string::npos) {
        oidPos += 8; // Skip past "oid": "
        size_t endPos = description.find("\"", oidPos);
        if (endPos != std::string::npos) {
            std::string oidStr = description.substr(oidPos, endPos - oidPos);
            Oid tableOid = static_cast<Oid>(std::stoul(oidStr));
            ::g_jit_table_oid = tableOid;
        }
    }
    scanContext = open_postgres_table(tableName.c_str());
}

void* PostgreSQLDataSource::getNext() {
    return nullptr;
}

PostgreSQLDataSource::~PostgreSQLDataSource() {
    if (scanContext) {
        RUNTIME_PGX_DEBUG("PostgreSQLDataSource", "Closing PostgreSQL table: " + tableName);
        close_postgres_table(scanContext);
    }
    RUNTIME_PGX_DEBUG("PostgreSQLDataSource", "PostgreSQL data source destroyed");
}

::runtime::DataSource* PostgreSQLDataSource::createFromDescription(::runtime::VarLen32 description) {
    // Convert VarLen32 to string
    std::string descStr(description.data(), description.getLen());
    
    RUNTIME_PGX_DEBUG("PostgreSQLDataSource", "Creating data source from description");
    // Return PostgreSQL-specific data source
    return new PostgreSQLDataSource(descStr);
}

} // namespace pgx_lower::compiler::runtime