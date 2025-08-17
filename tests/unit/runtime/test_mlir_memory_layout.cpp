#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

// Test the MLIR tuple layout structure to ensure it matches MLIR expectations
TEST(MLIRMemoryLayoutTest, TupleLayoutSize) {
    // MLIR expects exactly this layout (48 bytes total, no padding)
    struct MLIRTupleLayout {
        size_t numRows;           // Element 0: index type (8 bytes)
        size_t offset;            // Element 1: index (8 bytes)
        size_t validMultiplier;   // Element 2: index (8 bytes)  
        void* validBuffer;        // Element 3: !util.ref<i8> (8 bytes)
        void* dataBuffer;         // Element 4: !util.ref<i32> (8 bytes)
        void* varLenBuffer;       // Element 5: !util.ref<i8> (8 bytes)
    } __attribute__((packed));
    
    // Verify the structure is exactly 48 bytes (6 elements * 8 bytes each)
    EXPECT_EQ(sizeof(MLIRTupleLayout), 48);
    
    // Verify each field is at the correct offset
    EXPECT_EQ(offsetof(MLIRTupleLayout, numRows), 0);
    EXPECT_EQ(offsetof(MLIRTupleLayout, offset), 8);
    EXPECT_EQ(offsetof(MLIRTupleLayout, validMultiplier), 16);
    EXPECT_EQ(offsetof(MLIRTupleLayout, validBuffer), 24);
    EXPECT_EQ(offsetof(MLIRTupleLayout, dataBuffer), 32);
    EXPECT_EQ(offsetof(MLIRTupleLayout, varLenBuffer), 40);
}

// Test that size_t is the correct size for MLIR index type
TEST(MLIRMemoryLayoutTest, IndexTypeSize) {
    // MLIR index type on x86_64 should be 8 bytes
    EXPECT_EQ(sizeof(size_t), 8);
}

// Test pointer sizes match MLIR expectations
TEST(MLIRMemoryLayoutTest, PointerSize) {
    // MLIR !util.ref types are pointers, should be 8 bytes on x86_64
    EXPECT_EQ(sizeof(void*), 8);
}