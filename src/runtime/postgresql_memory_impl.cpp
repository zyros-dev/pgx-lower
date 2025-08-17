// PostgreSQL implementation of the memory abstraction layer
// This file contains all PostgreSQL-specific memory management code

#include "pgx_lower/runtime/memory_abstraction.h"
#include "execution/logging.h"
#include <memory>

extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
}

namespace pgx_lower {
namespace runtime {

// PostgreSQL-specific memory allocator implementation
class PostgreSQLMemoryAllocator : public MemoryAllocator {
public:
    void* allocate(size_t size) override {
        if (!isContextValid()) {
            PGX_ERROR("PostgreSQLMemoryAllocator: Invalid memory context");
            return nullptr;
        }
        
        void* ptr = nullptr;
        PG_TRY();
        {
            ptr = palloc(size);
        }
        PG_CATCH();
        {
            PGX_ERROR("PostgreSQLMemoryAllocator: palloc failed for size " + std::to_string(size));
            PG_RE_THROW();
        }
        PG_END_TRY();
        
        if (!ptr) {
            PGX_ERROR("PostgreSQLMemoryAllocator: Failed to allocate " + std::to_string(size) + " bytes");
        }
        return ptr;
    }
    
    void deallocate(void* ptr) override {
        if (ptr && isContextValid()) {
            PG_TRY();
            {
                pfree(ptr);
            }
            PG_CATCH();
            {
                PGX_WARNING("PostgreSQLMemoryAllocator: pfree failed");
                // Don't re-throw on deallocation failure
            }
            PG_END_TRY();
        }
    }
    
    bool isContextValid() const override {
        if (!CurrentMemoryContext) {
            return false;
        }
        
        // Verify context has valid methods (not corrupted)
        if (!CurrentMemoryContext->methods) {
            return false;
        }
        
        // Check for known valid context types
        if (CurrentMemoryContext->type != T_AllocSetContext && 
            CurrentMemoryContext->type != T_SlabContext && 
            CurrentMemoryContext->type != T_GenerationContext) {
            return false;
        }
        
        return true;
    }
    
    void* allocateWithFallback(size_t size, std::function<void*()> fallback) override {
        void* ptr = allocate(size);
        if (!ptr && fallback) {
            PGX_DEBUG("Primary allocation failed, trying fallback");
            ptr = fallback();
        }
        return ptr;
    }
};

// PostgreSQL-specific datum handler implementation
class PostgreSQLDatumHandler : public DatumHandler {
public:
    Datum copyDatum(Datum value, Oid typeOid, bool isNull) override {
        if (isNull) {
            return value;
        }
        
        switch (typeOid) {
            // Text types need deep copy
            case TEXTOID:
            case VARCHAROID:
            case BPCHAROID:
                return datumCopy(value, false, -1);
                
            // Scalar types are pass-by-value
            case INT2OID:
            case INT4OID:
            case INT8OID:
            case BOOLOID:
            case FLOAT4OID:
            case FLOAT8OID:
                return value;
                
            // Array types need deep copy
            case ANYARRAYOID:
            case TEXTARRAYOID:
            case INT4ARRAYOID:
                return datumCopy(value, false, -1);
                
            default:
                // Conservative: copy unknown types
                return datumCopy(value, false, -1);
        }
    }
    
    Datum createTextDatum(const char* str) override {
        if (!str) {
            return (Datum)0;
        }
        
        text* textval = nullptr;
        PG_TRY();
        {
            textval = cstring_to_text(str);
        }
        PG_CATCH();
        {
            PGX_ERROR("Failed to create text datum");
            PG_RE_THROW();
        }
        PG_END_TRY();
        
        return PointerGetDatum(textval);
    }
    
    int32_t datumToInt32(Datum value) override {
        return DatumGetInt32(value);
    }
    
    int64_t datumToInt64(Datum value) override {
        return DatumGetInt64(value);
    }
    
    bool datumToBool(Datum value) override {
        return DatumGetBool(value);
    }
    
    double datumToFloat8(Datum value) override {
        return DatumGetFloat8(value);
    }
    
    const char* datumToText(Datum value) override {
        if (!value) {
            return nullptr;
        }
        text* t = DatumGetTextP(value);
        return VARDATA(t);
    }
    
    Datum int32ToDatum(int32_t value) override {
        return Int32GetDatum(value);
    }
    
    Datum int64ToDatum(int64_t value) override {
        return Int64GetDatum(value);
    }
    
    Datum boolToDatum(bool value) override {
        return BoolGetDatum(value);
    }
    
    Datum float8ToDatum(double value) override {
        return Float8GetDatum(value);
    }
    
    Datum textToDatum(const char* value) override {
        return createTextDatum(value);
    }
};

// PostgreSQL-specific memory context manager implementation
class PostgreSQLMemoryContextManager : public MemoryContextManager {
public:
    MemoryContext switchContext(MemoryContext newContext) override {
        if (!newContext) {
            PGX_WARNING("Attempting to switch to null context");
            return getCurrentContext();
        }
        
        MemoryContext oldContext = CurrentMemoryContext;
        MemoryContextSwitchTo(static_cast<::MemoryContext>(newContext));
        return oldContext;
    }
    
    MemoryContext getCurrentContext() const override {
        return CurrentMemoryContext;
    }
    
    MemoryContext getTransactionContext() const override {
        return CurTransactionContext;
    }
    
    MemoryContext createTempContext(const char* name) override {
        if (!CurrentMemoryContext) {
            PGX_ERROR("Cannot create temp context without current context");
            return nullptr;
        }
        
        ::MemoryContext tempContext = nullptr;
        PG_TRY();
        {
            tempContext = AllocSetContextCreate(CurrentMemoryContext,
                                               name,
                                               ALLOCSET_DEFAULT_SIZES);
        }
        PG_CATCH();
        {
            PGX_ERROR("Failed to create temporary memory context");
            PG_RE_THROW();
        }
        PG_END_TRY();
        
        return tempContext;
    }
    
    void deleteContext(MemoryContext context) override {
        if (context && context != CurrentMemoryContext) {
            PG_TRY();
            {
                MemoryContextDelete(static_cast<::MemoryContext>(context));
            }
            PG_CATCH();
            {
                PGX_WARNING("Failed to delete memory context");
                // Don't re-throw on deletion failure
            }
            PG_END_TRY();
        }
    }
    
    void resetContext(MemoryContext context) override {
        if (context) {
            PG_TRY();
            {
                MemoryContextReset(static_cast<::MemoryContext>(context));
            }
            PG_CATCH();
            {
                PGX_WARNING("Failed to reset memory context");
                // Don't re-throw on reset failure
            }
            PG_END_TRY();
        }
    }
};

// Static instances
static std::unique_ptr<MemoryAllocator> g_allocator;
static std::unique_ptr<DatumHandler> g_datumHandler;
static std::unique_ptr<MemoryContextManager> g_contextManager;

// Factory implementation
MemoryAllocator* MemoryManagementFactory::getAllocator() {
    if (!g_allocator) {
        initialize(true);
    }
    return g_allocator.get();
}

DatumHandler* MemoryManagementFactory::getDatumHandler() {
    if (!g_datumHandler) {
        initialize(true);
    }
    return g_datumHandler.get();
}

MemoryContextManager* MemoryManagementFactory::getContextManager() {
    if (!g_contextManager) {
        initialize(true);
    }
    return g_contextManager.get();
}

void MemoryManagementFactory::initialize(bool usePostgreSQL) {
    if (usePostgreSQL) {
        g_allocator = std::make_unique<PostgreSQLMemoryAllocator>();
        g_datumHandler = std::make_unique<PostgreSQLDatumHandler>();
        g_contextManager = std::make_unique<PostgreSQLMemoryContextManager>();
        PGX_DEBUG("Initialized PostgreSQL memory management");
    } else {
        // Mock implementations would be initialized here for testing
        PGX_DEBUG("Mock memory management not yet implemented");
    }
}

void MemoryManagementFactory::cleanup() {
    g_allocator.reset();
    g_datumHandler.reset();
    g_contextManager.reset();
    PGX_DEBUG("Cleaned up memory management resources");
}

} // namespace runtime
} // namespace pgx_lower