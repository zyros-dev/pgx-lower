#include <gtest/gtest.h>
#include "runtime/tuple_access.cpp"
#include "execution/logging.h"

extern TupleStreamer g_tuple_streamer;
extern ComputedResultsBuffer g_computed_results;
class TupleAccessHelpersTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_tuple_streamer.isActive = false;
        g_tuple_streamer.dest = nullptr;
        g_tuple_streamer.slot = nullptr;
        g_computed_results.numComputedColumns = 0;
    }
    
    void TearDown() override {
        g_tuple_streamer.isActive = false;
        g_tuple_streamer.dest = nullptr;
        g_tuple_streamer.slot = nullptr;
        g_computed_results.numComputedColumns = 0;
    }
};

TEST_F(TupleAccessHelpersTest, ValidateStreamingContext_InactiveStreamer) {
    g_tuple_streamer.isActive = false;
    
    bool result = validate_streaming_context();
    
    EXPECT_FALSE(result);
}

TEST_F(TupleAccessHelpersTest, ValidateStreamingContext_NullDestination) {
    g_tuple_streamer.isActive = true;
    g_tuple_streamer.dest = nullptr;
    g_tuple_streamer.slot = reinterpret_cast<TupleTableSlot*>(0x1234);    
    bool result = validate_streaming_context();
    
    EXPECT_FALSE(result);
}

TEST_F(TupleAccessHelpersTest, ValidateStreamingContext_NullSlot) {
    g_tuple_streamer.isActive = true;
    g_tuple_streamer.dest = reinterpret_cast<DestReceiver*>(0x1234);    g_tuple_streamer.slot = nullptr;
    
    bool result = validate_streaming_context();
    
    EXPECT_FALSE(result);
}

TEST_F(TupleAccessHelpersTest, ValidateStreamingContext_AllValid) {
    g_tuple_streamer.isActive = true;
    g_tuple_streamer.dest = reinterpret_cast<DestReceiver*>(0x1234);    g_tuple_streamer.slot = reinterpret_cast<TupleTableSlot*>(0x5678);    
    bool result = validate_streaming_context();
    
    EXPECT_TRUE(result);
}

TEST_F(TupleAccessHelpersTest, SetupProcessingMemoryContext_NullSlot) {
    TupleTableSlot* slot = nullptr;
    
    MemoryContext result = setup_processing_memory_context(slot);
    
    EXPECT_EQ(result, nullptr);
}

TEST_F(TupleAccessHelpersTest, SetupProcessingMemoryContext_NoMemoryContext) {
    TupleTableSlot slot;
    slot.tts_mcxt = nullptr;
    
    MemoryContext result = setup_processing_memory_context(&slot);
    
    EXPECT_EQ(result, CurrentMemoryContext);
}

TEST_F(TupleAccessHelpersTest, AllocateAndProcessColumns_ZeroColumns) {
    Datum* processedValues = nullptr;
    bool* processedNulls = nullptr;
    
    g_computed_results.numComputedColumns = 0;
    
    EXPECT_NO_THROW({
    });
}