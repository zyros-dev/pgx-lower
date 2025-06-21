#include "interfaces/tuple_passthrough_manager.h"

#ifdef POSTGRESQL_EXTENSION
// Include PostgreSQL headers with proper macro definitions
extern "C" {
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "executor/tuptable.h"
#include "postgres.h"
#include "tcop/dest.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
}

#include "postgres/my_executor.h"

// Forward declarations for external state from my_executor.cpp
extern TupleScanContext* g_scan_context;
extern TupleStreamer g_tuple_streamer;
extern PostgreSQLTuplePassthrough g_current_tuple_passthrough;

// External function declarations from my_executor.cpp
extern "C" int64_t get_next_tuple();
extern "C" void* open_postgres_table(const char* tableName);
extern "C" int64_t read_next_tuple_from_table(void* tableHandle);
extern "C" void close_postgres_table(void* tableHandle);
extern "C" bool add_tuple_to_result(int64_t value);

TuplePassthroughManager& TuplePassthroughManager::getInstance() {
    static TuplePassthroughManager instance;
    return instance;
}

int64_t TuplePassthroughManager::readNextTuple(void* tableHandle) {
    // Delegate to existing implementation in my_executor.cpp
    return ::read_next_tuple_from_table(tableHandle);
}

bool TuplePassthroughManager::addTupleToResult() {
    // Delegate to existing implementation in my_executor.cpp
    return ::add_tuple_to_result(1); // Pass dummy value since it's ignored
}

void* TuplePassthroughManager::openTable(const char* tableName) {
    // Delegate to existing implementation in my_executor.cpp
    return ::open_postgres_table(tableName);
}

void TuplePassthroughManager::closeTable(void* tableHandle) {
    // Delegate to existing implementation in my_executor.cpp
    ::close_postgres_table(tableHandle);
}

int64_t TuplePassthroughManager::getNextTupleLegacy() {
    // Delegate to existing implementation in my_executor.cpp
    return ::get_next_tuple();
}

void TuplePassthroughManager::setStreamer(void* streamer) {
    m_streamer = static_cast<TupleStreamer*>(streamer);
}

void TuplePassthroughManager::setScanContext(void* context) {
    m_scanContext = static_cast<TupleScanContext*>(context);
}

void TuplePassthroughManager::setTuplePassthrough(void* passthrough) {
    m_tuplePassthrough = static_cast<PostgreSQLTuplePassthrough*>(passthrough);
}

#else
// Mock implementation for unit tests
TuplePassthroughManager& TuplePassthroughManager::getInstance() {
    static TuplePassthroughManager instance;
    return instance;
}

int64_t TuplePassthroughManager::readNextTuple(void* tableHandle) {
    return -1; // Mock implementation
}

bool TuplePassthroughManager::addTupleToResult() {
    return false; // Mock implementation
}

void* TuplePassthroughManager::openTable(const char* tableName) {
    return nullptr; // Mock implementation
}

void TuplePassthroughManager::closeTable(void* tableHandle) {
    // Mock implementation - do nothing
}

int64_t TuplePassthroughManager::getNextTupleLegacy() {
    return -1; // Mock implementation
}

void TuplePassthroughManager::setStreamer(void* streamer) {
    // Mock implementation - do nothing
}

void TuplePassthroughManager::setScanContext(void* context) {
    // Mock implementation - do nothing  
}

void TuplePassthroughManager::setTuplePassthrough(void* passthrough) {
    // Mock implementation - do nothing
}
#endif