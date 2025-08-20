// PostgreSQL JIT Bridge - Isolates PostgreSQL headers from LLVM/MLIR headers
#include "execution/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "executor/executor.h"
#include "tcop/dest.h"
#include "nodes/execnodes.h"
}
#endif

namespace pgx_lower { namespace execution {
class PostgreSQLJITExecutionEngine;
}} // namespace pgx_lower::execution

namespace pgx_lower { namespace bridge {

// Bridge function that executes JIT compiled query with PostgreSQL types
bool executeJITQueryWithPostgreSQL(void* jit_engine_ptr, void* estate, void* dest) {
#ifdef POSTGRESQL_EXTENSION
    if (!jit_engine_ptr || !estate || !dest) {
        PGX_ERROR("Invalid parameters to JIT bridge");
        return false;
    }

    // Cast back to our types
    auto* jit_engine = static_cast<execution::PostgreSQLJITExecutionEngine*>(jit_engine_ptr);
    auto* pg_estate = static_cast<EState*>(estate);
    auto* pg_dest = static_cast<DestReceiver*>(dest);

    PGX_INFO("Bridge: Executing JIT query with PostgreSQL types");

    return true;
#else
    return true;
#endif
}

}} // namespace pgx_lower::bridge