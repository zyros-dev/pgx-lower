#include "runtime/PostgreSQLDataSource.h"
#include "runtime/tuple_access.h"
#include <cstring>
#include <sstream>

// Note: Function signatures now defined in tuple_access.h

namespace pgx_lower::compiler::runtime {

PostgreSQLDataSource::PostgreSQLDataSource(const std::string& description) : scanContext(nullptr) {
    // Parse the JSON description to extract table name
    size_t tablePos = description.find("\"table\": \"");
    if (tablePos != std::string::npos) {
        tablePos += 10; // Skip past "table": "
        size_t endPos = description.find("\"", tablePos);
        if (endPos != std::string::npos) {
            tableName = description.substr(tablePos, endPos - tablePos);
        }
    }
    
    // Parse and set the table OID global variable
    size_t oidPos = description.find("\"oid\": \"");
    if (oidPos != std::string::npos) {
        oidPos += 8; // Skip past "oid": "
        size_t endPos = description.find("\"", oidPos);
        if (endPos != std::string::npos) {
            std::string oidStr = description.substr(oidPos, endPos - oidPos);
            Oid tableOid = static_cast<Oid>(std::stoul(oidStr));
            
            // Set the global variable for runtime table access
            ::g_jit_table_oid = tableOid;
        }
    }
    
    // Open the PostgreSQL table
    scanContext = open_postgres_table(tableName.c_str());
}

void PostgreSQLDataSource::iterate(bool parallel, std::vector<std::string> members, 
                                  const std::function<void(BatchView*)>& cb) {
    // PostgreSQL doesn't use Arrow batches, so we need a different approach
    // For now, this is a stub - the real implementation would need to
    // create a PostgreSQL-compatible BatchView or use a different iteration pattern
    
    // The actual iteration happens in ScanRefsTableLowering which expects
    // Arrow-style iteration. We'd need to either:
    // 1. Create a fake BatchView that wraps PostgreSQL tuples
    // 2. Modify ScanRefsTableLowering to handle PostgreSQL directly
}

PostgreSQLDataSource::~PostgreSQLDataSource() {
    if (scanContext) {
        close_postgres_table(scanContext);
    }
}

DataSource* PostgreSQLDataSource::createFromDescription(runtime::VarLen32 description) {
    // Convert VarLen32 to string
    std::string descStr(description.data(), description.getLen());
    
    // Return PostgreSQL-specific data source
    return new PostgreSQLDataSource(descStr);
}

} // namespace pgx_lower::compiler::runtime