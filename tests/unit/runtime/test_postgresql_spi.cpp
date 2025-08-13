#include <gtest/gtest.h>
#include "pgx_lower/runtime/postgresql_spi_stubs.h"
#include "execution/logging.h"
#include <vector>
#include <string>

// Mock PostgreSQL SPI structures for unit testing
#ifndef POSTGRESQL_EXTENSION

#define SPI_OK_CONNECT 1
#define SPI_OK_SELECT 5
#define SPI_ERROR_CONNECT -1

// Mock globals
static int SPI_result = 0;
static uint64_t SPI_processed = 0;
static struct SPITupleTable* SPI_tuptable = nullptr;
static bool SPI_connected = false;

// Mock structures
struct TupleDesc {
    int natts;
    const char* attrs[10];
};

struct HeapTuple {
    int32_t* data;
    int natts;
};

struct SPITupleTable {
    TupleDesc* tupdesc;
    HeapTuple** vals;
    uint64_t numvals;
};

// Mock SPI functions
extern "C" {
    int SPI_connect() {
        if (SPI_connected) {
            return SPI_ERROR_CONNECT;
        }
        SPI_connected = true;
        return SPI_OK_CONNECT;
    }
    
    int SPI_finish() {
        SPI_connected = false;
        SPI_tuptable = nullptr;
        SPI_processed = 0;
        return 0;
    }
    
    int SPI_exec(const char* query, int count) {
        // Simple mock: if query contains "test", return one row
        std::string q(query);
        if (q.find("test") != std::string::npos) {
            // Create mock result
            static TupleDesc tupdesc;
            tupdesc.natts = 1;
            tupdesc.attrs[0] = "id";
            
            static int32_t data = 1;
            static HeapTuple tuple;
            tuple.data = &data;
            tuple.natts = 1;
            
            static HeapTuple* tuples[1] = {&tuple};
            static SPITupleTable tuptable;
            tuptable.tupdesc = &tupdesc;
            tuptable.vals = tuples;
            tuptable.numvals = 1;
            
            SPI_tuptable = &tuptable;
            SPI_processed = 1;
            return SPI_OK_SELECT;
        }
        
        return -1;
    }
    
    bool heap_getattr_is_null = false;
    int32_t heap_getattr_value = 42;
    
    // Simplified heap_getattr for testing
    uint64_t heap_getattr(void* tuple, int attnum, void* tupdesc, bool* isnull) {
        *isnull = heap_getattr_is_null;
        return heap_getattr_value;
    }
    
    int32_t DatumGetInt32(uint64_t datum) {
        return static_cast<int32_t>(datum);
    }
}

// Cleanup function declaration
extern "C" void pg_cleanup_table_scans();

#endif // POSTGRESQL_EXTENSION

namespace pgx_lower::runtime::test {

class PostgreSQLSPITest : public ::testing::Test {
protected:
    void SetUp() override {
#ifndef POSTGRESQL_EXTENSION
        // Reset mock state
        SPI_connected = false;
        SPI_tuptable = nullptr;
        SPI_processed = 0;
        heap_getattr_is_null = false;
        heap_getattr_value = 42;
#endif
    }
    
    void TearDown() override {
#ifndef POSTGRESQL_EXTENSION
        // Cleanup
        pg_cleanup_table_scans();
        SPI_connected = false;
#endif
    }
};

TEST_F(PostgreSQLSPITest, OpenTableSuccess) {
#ifndef POSTGRESQL_EXTENSION
    void* handle = pg_table_open("test");
    EXPECT_NE(handle, nullptr);
    EXPECT_TRUE(SPI_connected);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, OpenTableNullName) {
    void* handle = pg_table_open(nullptr);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(PostgreSQLSPITest, OpenTableEmptyResult) {
#ifndef POSTGRESQL_EXTENSION
    void* handle = pg_table_open("nonexistent");
    // In our mock, this returns -1 from SPI_exec
    EXPECT_EQ(handle, nullptr);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, GetNextTupleSuccess) {
#ifndef POSTGRESQL_EXTENSION
    void* handle = pg_table_open("test");
    ASSERT_NE(handle, nullptr);
    
    int64_t tuple = pg_get_next_tuple(handle);
    EXPECT_NE(tuple, 0);
    
    // Second call should return 0 (no more tuples)
    tuple = pg_get_next_tuple(handle);
    EXPECT_EQ(tuple, 0);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, GetNextTupleNullHandle) {
    int64_t tuple = pg_get_next_tuple(nullptr);
    EXPECT_EQ(tuple, 0);
}

TEST_F(PostgreSQLSPITest, GetNextTupleInvalidHandle) {
    void* invalid_handle = reinterpret_cast<void*>(999);
    int64_t tuple = pg_get_next_tuple(invalid_handle);
    EXPECT_EQ(tuple, 0);
}

TEST_F(PostgreSQLSPITest, ExtractFieldSuccess) {
#ifndef POSTGRESQL_EXTENSION
    void* handle = pg_table_open("test");
    ASSERT_NE(handle, nullptr);
    
    int64_t tuple = pg_get_next_tuple(handle);
    ASSERT_NE(tuple, 0);
    
    int32_t value = pg_extract_field(reinterpret_cast<void*>(tuple), 0);
    EXPECT_EQ(value, 42); // Our mock value
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, ExtractFieldNullTuple) {
    int32_t value = pg_extract_field(nullptr, 0);
    EXPECT_EQ(value, 0);
}

TEST_F(PostgreSQLSPITest, ExtractFieldInvalidIndex) {
#ifndef POSTGRESQL_EXTENSION
    void* handle = pg_table_open("test");
    ASSERT_NE(handle, nullptr);
    
    int64_t tuple = pg_get_next_tuple(handle);
    ASSERT_NE(tuple, 0);
    
    int32_t value = pg_extract_field(reinterpret_cast<void*>(tuple), 5);
    EXPECT_EQ(value, 0);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, ExtractFieldNullValue) {
#ifndef POSTGRESQL_EXTENSION
    heap_getattr_is_null = true;
    
    void* handle = pg_table_open("test");
    ASSERT_NE(handle, nullptr);
    
    int64_t tuple = pg_get_next_tuple(handle);
    ASSERT_NE(tuple, 0);
    
    int32_t value = pg_extract_field(reinterpret_cast<void*>(tuple), 0);
    EXPECT_EQ(value, 0);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, CompleteTableScan) {
#ifndef POSTGRESQL_EXTENSION
    // Open table
    void* handle = pg_table_open("test");
    ASSERT_NE(handle, nullptr);
    
    // Read all tuples
    std::vector<int32_t> values;
    int64_t tuple;
    while ((tuple = pg_get_next_tuple(handle)) != 0) {
        int32_t value = pg_extract_field(reinterpret_cast<void*>(tuple), 0);
        values.push_back(value);
    }
    
    // Should have read one tuple with value 42
    EXPECT_EQ(values.size(), 1);
    EXPECT_EQ(values[0], 42);
    
    // Cleanup
    pg_cleanup_table_scans();
    EXPECT_FALSE(SPI_connected);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

TEST_F(PostgreSQLSPITest, MultipleTableScans) {
#ifndef POSTGRESQL_EXTENSION
    // Open first table
    void* handle1 = pg_table_open("test");
    ASSERT_NE(handle1, nullptr);
    
    // Try to open second table (should work with our implementation)
    void* handle2 = pg_table_open("test");
    ASSERT_NE(handle2, nullptr);
    EXPECT_NE(handle1, handle2);
    
    // Both should be able to read tuples
    int64_t tuple1 = pg_get_next_tuple(handle1);
    EXPECT_NE(tuple1, 0);
    
    int64_t tuple2 = pg_get_next_tuple(handle2);
    EXPECT_NE(tuple2, 0);
#else
    GTEST_SKIP() << "Test only valid in mock environment";
#endif
}

} // namespace pgx_lower::runtime::test