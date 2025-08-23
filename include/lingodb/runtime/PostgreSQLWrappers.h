#ifndef PGX_LOWER_RUNTIME_POSTGRESQLWRAPPERS_H
#define PGX_LOWER_RUNTIME_POSTGRESQLWRAPPERS_H

#include "lingodb/runtime/helpers.h"
#include <cstdint>

namespace pgx_lower::compiler::runtime {

// PostgreSQL table handle wrapper
struct PostgreSQLTableHandle {
    void* scanContext;  // Points to TupleScanContext
    const char* tableName;
    int64_t currentRow;
};

// Function wrappers that will be called from MLIR
class PostgreSQLDataSource {
public:
    // Open a PostgreSQL table for scanning
    static PostgreSQLTableHandle* openTable(VarLen32 description);
    
    // Check if there are more tuples
    static bool hasNextTuple(PostgreSQLTableHandle* handle);
    
    // Read the next tuple
    static void* readNextTuple(PostgreSQLTableHandle* handle);
    
    // Get field value from tuple
    static int32_t getIntField(void* tuple, int32_t fieldIndex);
    static const char* getTextField(void* tuple, int32_t fieldIndex);
    static double getNumericField(void* tuple, int32_t fieldIndex);
    
    // Close table
    static void closeTable(PostgreSQLTableHandle* handle);
};

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_POSTGRESQLWRAPPERS_H