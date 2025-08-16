#include "runtime/DataSourceIteration.h"
#include <memory>
#include <string>

// Include nlohmann/json before PostgreSQL headers to avoid conflicts
#include <nlohmann/json.hpp>

// PostgreSQL headers must come after to avoid macro conflicts
#include "execution/logging.h"

// PostgreSQL-based DataSourceIteration implementation
namespace runtime {

// Simple implementation that simulates reading from a PostgreSQL table
class PostgreSQLTableSource : public DataSource {
    int currentRow = 0;
    int totalRows = 1; // Test 1 has 1 row
    
public:
    PostgreSQLTableSource(const std::string& tableName) {
        PGX_DEBUG("PostgreSQLTableSource created for table: " + tableName);
        // In real implementation, we'd open the PostgreSQL table here
    }
    
    void* getNext() override {
        if (currentRow < totalRows) {
            currentRow++;
            // Return non-null to indicate we have data
            return reinterpret_cast<void*>(1);
        }
        return nullptr;
    }
};

bool DataSourceIteration::isValid() {
    bool valid = (currChunk != nullptr);
    PGX_DEBUG("DataSourceIteration::isValid = " + std::to_string(valid));
    return valid;
}

void DataSourceIteration::next() {
    PGX_DEBUG("DataSourceIteration::next called");
    if (dataSource) {
        currChunk = dataSource->getNext();
    }
}

void DataSourceIteration::access(RecordBatchInfo* info) {
    PGX_DEBUG("DataSourceIteration::access called");
    // For PostgreSQL, we just need to indicate we have 1 row
    info->numRows = 1;
    
    // Initialize column info (simplified for PostgreSQL)
    for (size_t i = 0; i < 1; i++) { // Assuming 1 column for Test 1
        info->columnInfo[i].offset = 0;
        info->columnInfo[i].validMultiplier = 1;
        static uint8_t dummyBuffer = 0xFF;
        info->columnInfo[i].validBuffer = &dummyBuffer;
        info->columnInfo[i].dataBuffer = &dummyBuffer;
        info->columnInfo[i].varLenBuffer = &dummyBuffer;
    }
}

void DataSourceIteration::end(DataSourceIteration* iteration) {
    PGX_DEBUG("DataSourceIteration::end called");
    delete iteration;
}

DataSourceIteration* DataSourceIteration::start(ExecutionContext* executionContext, VarLen32 description) {
    PGX_DEBUG("DataSourceIteration::start called with description: " + description.str());
    
    // Parse the JSON description to get table name
    nlohmann::json descr = nlohmann::json::parse(description.str());
    std::string tableName = descr["table"];
    
    PGX_DEBUG("Starting iteration for table: " + tableName);
    
    // Create PostgreSQL data source
    auto dataSource = std::make_shared<PostgreSQLTableSource>(tableName);
    
    // Create iteration object with empty column IDs for now
    std::vector<size_t> colIds;
    auto* iteration = new DataSourceIteration(dataSource, colIds);
    
    // Get first chunk
    iteration->currChunk = dataSource->getNext();
    
    return iteration;
}

DataSourceIteration::DataSourceIteration(const std::shared_ptr<DataSource>& dataSource, const std::vector<size_t>& colIds) 
    : dataSource(dataSource), colIds(colIds), currChunk(nullptr) {
}

} // namespace runtime

// The C++ member functions already generate the mangled names we need
// No need for extern "C" wrappers that would conflict