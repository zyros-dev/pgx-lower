// C Interface Layer for MLIR-PostgreSQL Integration
// This file provides pure C functions that MLIR can call
// All PostgreSQL-specific logic is delegated to core classes

#include "interfaces/tuple_passthrough_manager.h"

#ifdef POSTGRESQL_EXTENSION

extern "C" {

int64_t read_next_tuple_from_table(void* tableHandle) {
    return TuplePassthroughManager::getInstance().readNextTuple(tableHandle);
}

bool add_tuple_to_result(int64_t value) {
    return TuplePassthroughManager::getInstance().addTupleToResult();
}

void* open_postgres_table(const char* tableName) {
    return TuplePassthroughManager::getInstance().openTable(tableName);
}

void close_postgres_table(void* tableHandle) {
    TuplePassthroughManager::getInstance().closeTable(tableHandle);
}

int64_t get_next_tuple() {
    return TuplePassthroughManager::getInstance().getNextTupleLegacy();
}

} // extern "C"

#endif // POSTGRESQL_EXTENSION