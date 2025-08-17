// Memory Abstraction Layer
// Provides clean separation between MLIR execution and PostgreSQL memory management
// This interface allows MLIR code to work with memory without direct PostgreSQL dependencies

#ifndef PGX_LOWER_RUNTIME_MEMORY_ABSTRACTION_H
#define PGX_LOWER_RUNTIME_MEMORY_ABSTRACTION_H

#include <cstddef>
#include <cstdint>
#include <functional>

namespace pgx_lower {
namespace runtime {

// Forward declarations to avoid PostgreSQL header dependencies
using MemoryContext = void*;
using Datum = uintptr_t;
using Oid = uint32_t;

// Memory allocation interface - no PostgreSQL types exposed
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;
    
    // Allocate memory in the appropriate context
    virtual void* allocate(size_t size) = 0;
    
    // Deallocate memory
    virtual void deallocate(void* ptr) = 0;
    
    // Check if allocation is safe
    virtual bool isContextValid() const = 0;
    
    // Allocate with error handling
    virtual void* allocateWithFallback(size_t size, std::function<void*()> fallback) = 0;
};

// Datum handling interface - abstracts PostgreSQL datum operations
class DatumHandler {
public:
    virtual ~DatumHandler() = default;
    
    // Copy a datum based on its type
    virtual Datum copyDatum(Datum value, Oid typeOid, bool isNull) = 0;
    
    // Create a text datum from a C string
    virtual Datum createTextDatum(const char* str) = 0;
    
    // Convert datum to specific types
    virtual int32_t datumToInt32(Datum value) = 0;
    virtual int64_t datumToInt64(Datum value) = 0;
    virtual bool datumToBool(Datum value) = 0;
    virtual double datumToFloat8(Datum value) = 0;
    virtual const char* datumToText(Datum value) = 0;
    
    // Create datums from specific types
    virtual Datum int32ToDatum(int32_t value) = 0;
    virtual Datum int64ToDatum(int64_t value) = 0;
    virtual Datum boolToDatum(bool value) = 0;
    virtual Datum float8ToDatum(double value) = 0;
    virtual Datum textToDatum(const char* value) = 0;
};

// Memory context management interface
class MemoryContextManager {
public:
    virtual ~MemoryContextManager() = default;
    
    // Switch to a different memory context
    virtual MemoryContext switchContext(MemoryContext newContext) = 0;
    
    // Get current memory context
    virtual MemoryContext getCurrentContext() const = 0;
    
    // Get transaction-level context
    virtual MemoryContext getTransactionContext() const = 0;
    
    // Create a temporary context
    virtual MemoryContext createTempContext(const char* name) = 0;
    
    // Delete a context
    virtual void deleteContext(MemoryContext context) = 0;
    
    // Reset a context (free all allocations but keep context)
    virtual void resetContext(MemoryContext context) = 0;
};

// RAII wrapper for memory context switching
class MemoryContextSwitcher {
private:
    MemoryContextManager* manager;
    MemoryContext oldContext;
    bool switched;
    
public:
    MemoryContextSwitcher(MemoryContextManager* mgr, MemoryContext newContext)
        : manager(mgr), oldContext(nullptr), switched(false) {
        if (manager && newContext) {
            oldContext = manager->switchContext(newContext);
            switched = true;
        }
    }
    
    ~MemoryContextSwitcher() {
        if (switched && manager && oldContext) {
            manager->switchContext(oldContext);
        }
    }
    
    // Disable copy/move to ensure RAII semantics
    MemoryContextSwitcher(const MemoryContextSwitcher&) = delete;
    MemoryContextSwitcher& operator=(const MemoryContextSwitcher&) = delete;
    MemoryContextSwitcher(MemoryContextSwitcher&&) = delete;
    MemoryContextSwitcher& operator=(MemoryContextSwitcher&&) = delete;
    
    bool isActive() const { return switched; }
};

// Factory for creating memory management implementations
class MemoryManagementFactory {
public:
    // Get the allocator instance
    static MemoryAllocator* getAllocator();
    
    // Get the datum handler instance
    static DatumHandler* getDatumHandler();
    
    // Get the context manager instance
    static MemoryContextManager* getContextManager();
    
    // Initialize with specific implementation (PostgreSQL or mock for testing)
    static void initialize(bool usePostgreSQL = true);
    
    // Cleanup resources
    static void cleanup();
};

// Helper functions for common operations
namespace MemoryHelpers {
    // Allocate and check for failure
    template<typename T>
    T* allocateChecked(size_t count = 1) {
        auto* allocator = MemoryManagementFactory::getAllocator();
        if (!allocator || !allocator->isContextValid()) {
            return nullptr;
        }
        
        void* ptr = allocator->allocate(sizeof(T) * count);
        return static_cast<T*>(ptr);
    }
    
    // Safe deallocation
    template<typename T>
    void deallocateSafe(T* ptr) {
        if (ptr) {
            auto* allocator = MemoryManagementFactory::getAllocator();
            if (allocator) {
                allocator->deallocate(ptr);
            }
        }
    }
    
    // RAII wrapper for allocated memory
    template<typename T>
    class ScopedAllocation {
    private:
        T* ptr;
        MemoryAllocator* allocator;
        
    public:
        explicit ScopedAllocation(size_t count = 1)
            : ptr(nullptr), allocator(MemoryManagementFactory::getAllocator()) {
            if (allocator && allocator->isContextValid()) {
                ptr = static_cast<T*>(allocator->allocate(sizeof(T) * count));
            }
        }
        
        ~ScopedAllocation() {
            if (ptr && allocator) {
                allocator->deallocate(ptr);
            }
        }
        
        // Disable copy, enable move
        ScopedAllocation(const ScopedAllocation&) = delete;
        ScopedAllocation& operator=(const ScopedAllocation&) = delete;
        
        ScopedAllocation(ScopedAllocation&& other) noexcept
            : ptr(other.ptr), allocator(other.allocator) {
            other.ptr = nullptr;
            other.allocator = nullptr;
        }
        
        ScopedAllocation& operator=(ScopedAllocation&& other) noexcept {
            if (this != &other) {
                if (ptr && allocator) {
                    allocator->deallocate(ptr);
                }
                ptr = other.ptr;
                allocator = other.allocator;
                other.ptr = nullptr;
                other.allocator = nullptr;
            }
            return *this;
        }
        
        T* get() { return ptr; }
        const T* get() const { return ptr; }
        T* release() {
            T* tmp = ptr;
            ptr = nullptr;
            return tmp;
        }
        bool isValid() const { return ptr != nullptr; }
        
        T& operator*() { return *ptr; }
        const T& operator*() const { return *ptr; }
        T* operator->() { return ptr; }
        const T* operator->() const { return ptr; }
    };
}

} // namespace runtime
} // namespace pgx_lower

#endif // PGX_LOWER_RUNTIME_MEMORY_ABSTRACTION_H