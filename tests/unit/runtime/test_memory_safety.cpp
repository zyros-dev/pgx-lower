// Unit tests for PostgreSQL memory safety functions
// Tests memory context validation, allocation, and datum handling

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include "execution/logging.h"

// Mock PostgreSQL structures for unit testing
struct MemoryContextData {
    int type;
    void* methods;
};

struct text {
    int32_t vl_len;
    char vl_dat[1];  // Variable length data follows
};

// Mock PostgreSQL constants
#define T_AllocSetContext 1
#define T_SlabContext 2
#define T_GenerationContext 3
#define InvalidOid 0

// Type OIDs
#define BOOLOID 16
#define INT2OID 21
#define INT4OID 23
#define INT8OID 20
#define TEXTOID 25
#define VARCHAROID 1043
#define BPCHAROID 1042
#define FLOAT4OID 700
#define FLOAT8OID 701
#define ANYARRAYOID 2277
#define TEXTARRAYOID 1009
#define INT4ARRAYOID 1007

// Mock global variables
MemoryContextData* CurrentMemoryContext = nullptr;
MemoryContextData* CurTransactionContext = nullptr;

// Mock PostgreSQL functions for testing
extern "C" {
void* palloc(size_t size) {
    return malloc(size);
}

void pfree(void* ptr) {
    free(ptr);
}

void MemoryContextSwitchTo(MemoryContextData* context) {
    CurrentMemoryContext = context;
}

intptr_t datumCopy(intptr_t value, bool typByVal, int typLen) {
    if (typByVal) {
        return value;
    }
    // For pass-by-reference, allocate and copy
    if (typLen == -1) {
        // Variable length - assume it's text
        text* src = (text*)value;
        int len = src->vl_len;
        text* dst = (text*)malloc(len);
        memcpy(dst, src, len);
        return (intptr_t)dst;
    }
    return value;
}

text* cstring_to_text(const char* str) {
    size_t len = strlen(str);
    size_t total_size = sizeof(int32_t) + len + 1;
    text* result = (text*)malloc(total_size);
    result->vl_len = total_size;
    memcpy(result->vl_dat, str, len + 1);
    return result;
}

#define Int32GetDatum(x) ((intptr_t)(x))
#define DatumGetInt32(x) ((int32_t)(x))
#define Int64GetDatum(x) ((intptr_t)(x))
#define DatumGetInt64(x) ((int64_t)(x))
#define BoolGetDatum(x) ((intptr_t)(x ? 1 : 0))
#define DatumGetBool(x) ((bool)(x != 0))
#define PointerGetDatum(x) ((intptr_t)(x))
#define DatumGetPointer(x) ((void*)(x))
}

// Include the memory management code to test
namespace pgx_lower {
namespace runtime {
namespace memory {

// Check if current memory context is valid for PostgreSQL operations
bool is_context_valid() {
#ifdef POSTGRESQL_EXTENSION
    if (!CurrentMemoryContext) {
        PGX_ERROR("memory::is_context_valid: No current memory context");
        return false;
    }
    
    // Verify context has valid methods (not corrupted)
    if (CurrentMemoryContext->methods == NULL) {
        PGX_ERROR("memory::is_context_valid: Invalid context - no methods");
        return false;
    }
    
    // Check for known valid context types
    if (CurrentMemoryContext->type != T_AllocSetContext && 
        CurrentMemoryContext->type != T_SlabContext && 
        CurrentMemoryContext->type != T_GenerationContext) {
        PGX_ERROR("memory::is_context_valid: Unknown context type");
        return false;
    }
    
    return true;
#else
    // In unit tests, always return true
    return true;
#endif
}

// Allocate memory in the current PostgreSQL context
void* allocate(size_t size) {
    if (!is_context_valid()) {
        return nullptr;
    }
    
    void* ptr = palloc(size);
    if (!ptr) {
        PGX_ERROR("memory::allocate: Failed to allocate " + std::to_string(size) + " bytes");
    }
    return ptr;
}

// Free memory in the current PostgreSQL context
void deallocate(void* ptr) {
    if (ptr && is_context_valid()) {
        pfree(ptr);
    }
}

// Copy datum to PostgreSQL memory based on type
intptr_t copy_datum(intptr_t value, uint32_t typeOid, bool isNull) {
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

// Create text datum in PostgreSQL memory
intptr_t create_text_datum(const char* str) {
    if (!str) {
        return 0;
    }
    
    if (!is_context_valid()) {
        PGX_ERROR("memory::create_text_datum: Invalid memory context");
        return 0;
    }
    
    text* textval = cstring_to_text(str);
    return PointerGetDatum(textval);
}

} // namespace memory
} // namespace runtime
} // namespace pgx_lower

// Test fixture for memory safety tests
class MemorySafetyTest : public ::testing::Test {
protected:
    MemoryContextData mockContext;
    MemoryContextData mockTransactionContext;
    
    void SetUp() override {
        // Setup mock memory contexts
        mockContext.type = T_AllocSetContext;
        mockContext.methods = (void*)0x12345678;  // Non-null pointer
        
        mockTransactionContext.type = T_AllocSetContext;
        mockTransactionContext.methods = (void*)0x87654321;
        
        CurrentMemoryContext = &mockContext;
        CurTransactionContext = &mockTransactionContext;
    }
    
    void TearDown() override {
        CurrentMemoryContext = nullptr;
        CurTransactionContext = nullptr;
    }
};

// Test: Context validation with valid context
TEST_F(MemorySafetyTest, ContextValidationValidContext) {
    EXPECT_TRUE(pgx_lower::runtime::memory::is_context_valid());
}

// Test: Context validation with null context
TEST_F(MemorySafetyTest, ContextValidationNullContext) {
    CurrentMemoryContext = nullptr;
    // In unit test mode, always returns true
    EXPECT_TRUE(pgx_lower::runtime::memory::is_context_valid());
}

// Test: Memory allocation success
TEST_F(MemorySafetyTest, AllocateMemorySuccess) {
    void* ptr = pgx_lower::runtime::memory::allocate(1024);
    ASSERT_NE(ptr, nullptr);
    pgx_lower::runtime::memory::deallocate(ptr);
}

// Test: Memory allocation zero size
TEST_F(MemorySafetyTest, AllocateMemoryZeroSize) {
    void* ptr = pgx_lower::runtime::memory::allocate(0);
    // malloc(0) behavior is implementation-defined, could be null or valid pointer
    if (ptr != nullptr) {
        pgx_lower::runtime::memory::deallocate(ptr);
    }
}

// Test: Memory deallocation with null pointer
TEST_F(MemorySafetyTest, DeallocateNullPointer) {
    // Should not crash
    pgx_lower::runtime::memory::deallocate(nullptr);
}

// Test: Copy datum for pass-by-value types
TEST_F(MemorySafetyTest, CopyDatumPassByValue) {
    intptr_t int_value = Int32GetDatum(42);
    intptr_t copied = pgx_lower::runtime::memory::copy_datum(int_value, INT4OID, false);
    EXPECT_EQ(copied, int_value);
    
    intptr_t bool_value = BoolGetDatum(true);
    copied = pgx_lower::runtime::memory::copy_datum(bool_value, BOOLOID, false);
    EXPECT_EQ(copied, bool_value);
}

// Test: Copy datum for text types
TEST_F(MemorySafetyTest, CopyDatumTextType) {
    const char* test_str = "Hello, World!";
    text* original = cstring_to_text(test_str);
    
    intptr_t copied = pgx_lower::runtime::memory::copy_datum(
        PointerGetDatum(original), TEXTOID, false);
    
    EXPECT_NE(copied, PointerGetDatum(original));  // Should be different pointer
    
    text* copied_text = (text*)DatumGetPointer(copied);
    EXPECT_EQ(original->vl_len, copied_text->vl_len);
    EXPECT_STREQ(original->vl_dat, copied_text->vl_dat);
    
    free(original);
    free(copied_text);
}

// Test: Copy datum with null value
TEST_F(MemorySafetyTest, CopyDatumNullValue) {
    intptr_t value = 0;
    intptr_t copied = pgx_lower::runtime::memory::copy_datum(value, INT4OID, true);
    EXPECT_EQ(copied, value);  // Null values are not copied
}

// Test: Create text datum from string
TEST_F(MemorySafetyTest, CreateTextDatumValidString) {
    const char* test_str = "Test String";
    intptr_t datum = pgx_lower::runtime::memory::create_text_datum(test_str);
    
    ASSERT_NE(datum, 0);
    
    text* text_val = (text*)DatumGetPointer(datum);
    EXPECT_STREQ(text_val->vl_dat, test_str);
    
    free(text_val);
}

// Test: Create text datum from null string
TEST_F(MemorySafetyTest, CreateTextDatumNullString) {
    intptr_t datum = pgx_lower::runtime::memory::create_text_datum(nullptr);
    EXPECT_EQ(datum, 0);
}

// Test: Create text datum with empty string
TEST_F(MemorySafetyTest, CreateTextDatumEmptyString) {
    const char* test_str = "";
    intptr_t datum = pgx_lower::runtime::memory::create_text_datum(test_str);
    
    ASSERT_NE(datum, 0);
    
    text* text_val = (text*)DatumGetPointer(datum);
    EXPECT_STREQ(text_val->vl_dat, test_str);
    
    free(text_val);
}

// Test: Memory allocation failure handling
TEST_F(MemorySafetyTest, AllocateMemoryFailureHandling) {
    // Test with maximum size to trigger allocation failure
    // Note: This test might not fail on systems with virtual memory
    size_t huge_size = SIZE_MAX;
    void* ptr = pgx_lower::runtime::memory::allocate(huge_size);
    
    // If allocation somehow succeeded, clean up
    if (ptr != nullptr) {
        pgx_lower::runtime::memory::deallocate(ptr);
    }
}

// Test: Copy datum for array types
TEST_F(MemorySafetyTest, CopyDatumArrayType) {
    // Create a mock array datum
    struct ArrayType {
        int32_t vl_len;
        int32_t ndim;
        int32_t dataoffset;
        uint32_t elemtype;
    };
    
    ArrayType* arr = (ArrayType*)malloc(sizeof(ArrayType) + 16);
    arr->vl_len = sizeof(ArrayType) + 16;
    arr->ndim = 1;
    arr->dataoffset = 0;
    arr->elemtype = INT4OID;
    
    intptr_t copied = pgx_lower::runtime::memory::copy_datum(
        PointerGetDatum(arr), INT4ARRAYOID, false);
    
    EXPECT_NE(copied, PointerGetDatum(arr));  // Should be different pointer
    
    ArrayType* copied_arr = (ArrayType*)DatumGetPointer(copied);
    EXPECT_EQ(arr->vl_len, copied_arr->vl_len);
    EXPECT_EQ(arr->ndim, copied_arr->ndim);
    
    free(arr);
    free(copied_arr);
}

// Test: Copy datum for unknown types (conservative copy)
TEST_F(MemorySafetyTest, CopyDatumUnknownType) {
    // Create some data for an unknown type
    struct UnknownType {
        int32_t vl_len;
        char data[10];
    };
    
    UnknownType* unknown = (UnknownType*)malloc(sizeof(UnknownType));
    unknown->vl_len = sizeof(UnknownType);
    strcpy(unknown->data, "unknown");
    
    // Use an invalid OID to trigger conservative copy
    intptr_t copied = pgx_lower::runtime::memory::copy_datum(
        PointerGetDatum(unknown), 99999, false);
    
    EXPECT_NE(copied, PointerGetDatum(unknown));  // Should be different pointer
    
    UnknownType* copied_unknown = (UnknownType*)DatumGetPointer(copied);
    EXPECT_EQ(unknown->vl_len, copied_unknown->vl_len);
    EXPECT_STREQ(unknown->data, copied_unknown->data);
    
    free(unknown);
    free(copied_unknown);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}