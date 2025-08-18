
#include "execution/logging.h"
#include <cstdint>
#include <cstring>

extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"  // For MaxAllocSize
#include "varatt.h"     // For VARHDRSZ
}

namespace pgx_lower {
namespace runtime {
namespace memory {

bool is_context_valid() {
#ifdef POSTGRESQL_EXTENSION
    if (!CurrentMemoryContext) {
        PGX_ERROR("memory::is_context_valid: No current memory context");
        return false;
    }
    
    if (CurrentMemoryContext->methods == NULL) {
        PGX_ERROR("memory::is_context_valid: Invalid context - no methods");
        return false;
    }
    
    if (CurrentMemoryContext->type != T_AllocSetContext && 
        CurrentMemoryContext->type != T_SlabContext && 
        CurrentMemoryContext->type != T_GenerationContext) {
        PGX_ERROR("memory::is_context_valid: Unknown context type");
        return false;
    }
    
    return true;
#else
    return true;
#endif
}

void* allocate(size_t size) {
    if (!is_context_valid()) {
        PGX_ERROR("memory::allocate: Invalid context, cannot allocate " + std::to_string(size) + " bytes");
        return nullptr;
    }
    
    // Check for zero or overflow size
    if (size == 0) {
        PGX_WARNING("memory::allocate: Zero size allocation requested");
        return nullptr;
    }
    
    if (size > MaxAllocSize) {
        PGX_ERROR("memory::allocate: Size " + std::to_string(size) + " exceeds maximum");
        return nullptr;
    }
    
    void* ptr = nullptr;
    PG_TRY();
    {
        ptr = palloc(size);
    }
    PG_CATCH();
    {
        PGX_ERROR("memory::allocate: palloc threw exception for size " + std::to_string(size));
        ptr = nullptr;
        // Don't re-throw to allow graceful degradation
    }
    PG_END_TRY();
    
    if (!ptr) {
        PGX_ERROR("memory::allocate: Failed to allocate " + std::to_string(size) + " bytes");
    }
    return ptr;
}

// Free memory in the current PostgreSQL context with error handling
void deallocate(void* ptr) {
    if (!ptr) {
        return;  // Null pointer is a no-op
    }
    
    if (!is_context_valid()) {
        PGX_WARNING("memory::deallocate: Invalid context, cannot deallocate");
        return;
    }
    
    PG_TRY();
    {
        pfree(ptr);
    }
    PG_CATCH();
    {
        PGX_WARNING("memory::deallocate: pfree threw exception");
        // Don't re-throw on deallocation failure
    }
    PG_END_TRY();
}

// Switch to a specific memory context
class ContextSwitcher {
private:
    MemoryContext oldContext;
    bool switched;
    
public:
    explicit ContextSwitcher(MemoryContext newContext) 
        : oldContext(nullptr), switched(false) {
        if (newContext && is_context_valid()) {
            oldContext = CurrentMemoryContext;
            MemoryContextSwitchTo(newContext);
            switched = true;
        }
    }
    
    ~ContextSwitcher() {
        if (switched && oldContext) {
            MemoryContextSwitchTo(oldContext);
        }
    }
    
    bool is_switched() const { return switched; }
};

// Copy datum to PostgreSQL memory based on type
Datum copy_datum(Datum value, Oid typeOid, bool isNull) {
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

// Create text datum in PostgreSQL memory with comprehensive error handling
Datum create_text_datum(const char* str) {
    if (!str) {
        PGX_DEBUG("memory::create_text_datum: Null string provided");
        return (Datum)0;
    }
    
    if (!is_context_valid()) {
        PGX_ERROR("memory::create_text_datum: Invalid memory context");
        return (Datum)0;
    }
    
    // Check string length to prevent overflow
    size_t len = strlen(str);
    if (len > MaxAllocSize - VARHDRSZ) {
        PGX_ERROR("memory::create_text_datum: String too long (" + std::to_string(len) + " bytes)");
        return (Datum)0;
    }
    
    // Use transaction context for stability
    ContextSwitcher switcher(CurTransactionContext);
    if (!switcher.is_switched()) {
        PGX_ERROR("memory::create_text_datum: Failed to switch context");
        return (Datum)0;
    }
    
    text* textval = nullptr;
    PG_TRY();
    {
        textval = cstring_to_text(str);
    }
    PG_CATCH();
    {
        PGX_ERROR("memory::create_text_datum: cstring_to_text failed");
        textval = nullptr;
    }
    PG_END_TRY();
    
    if (!textval) {
        PGX_ERROR("memory::create_text_datum: Failed to create text datum");
        return (Datum)0;
    }
    
    return PointerGetDatum(textval);
}

} // namespace memory
} // namespace runtime
} // namespace pgx_lower

// C interface for runtime functions
extern "C" {

bool pgx_memory_is_valid() {
    return pgx_lower::runtime::memory::is_context_valid();
}

void* pgx_memory_allocate(size_t size) {
    return pgx_lower::runtime::memory::allocate(size);
}

void pgx_memory_deallocate(void* ptr) {
    pgx_lower::runtime::memory::deallocate(ptr);
}

} // extern "C"