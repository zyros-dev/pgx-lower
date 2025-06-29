#pragma once

#include <cstdint>

// Forward declarations for PostgreSQL types
struct TupleScanContext;
struct PostgreSQLTuplePassthrough;
struct TupleStreamer;

// Core class that manages the tuple passthrough logic
// Separates PostgreSQL-specific implementation from C interface
class TuplePassthroughManager {
   public:
    static TuplePassthroughManager& getInstance();

    // C Interface implementations
    int64_t readNextTuple(void* tableHandle);
    bool addTupleToResult();
    void* openTable(const char* tableName);
    void closeTable(void* tableHandle);
    int64_t getNextTupleLegacy();

    // Internal management (use void* for compatibility with unit tests)
    void setStreamer(void* streamer);
    void setScanContext(void* context);
    void setTuplePassthrough(void* passthrough);

   private:
    TuplePassthroughManager() = default;

    // References to global state (managed by my_executor.cpp)
    class TupleStreamer* m_streamer = nullptr;
    class TupleScanContext* m_scanContext = nullptr;
    class PostgreSQLTuplePassthrough* m_tuplePassthrough = nullptr;
};