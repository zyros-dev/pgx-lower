#include "runtime/metadata.h"

using namespace runtime;

// Stub implementations matching exact interface from metadata.h

const std::optional<size_t>& ColumnMetaData::getDistinctValues() const {
    return distinctValues;
}

void ColumnMetaData::setDistinctValues(const std::optional<size_t>& dv) {
    distinctValues = dv;
}

void ColumnMetaData::setColumnType(const ColumnType& type) {
    this->columnType = type;
}

const ColumnType& ColumnMetaData::getColumnType() const {
    return columnType;
}

// TableMetaData methods - matching the inline implementations in header
const std::vector<std::string>& TableMetaData::getOrderedColumns() const {
    return orderedColumns;
}

// Static methods - stub implementations 
std::shared_ptr<TableMetaData> TableMetaData::deserialize(std::string json) {
    // TODO: Replace JSON deserialization with PostgreSQL catalog queries
    return std::make_shared<TableMetaData>();
}

std::string TableMetaData::serialize(bool serializeSample) const {
    // TODO: Replace JSON serialization with PostgreSQL metadata format
    // For now, return empty JSON
    return "{}";
}

std::shared_ptr<TableMetaData> TableMetaData::create(const std::string& json, const std::string& name, void* sample) {
    // TODO: Create from PostgreSQL catalog instead of JSON
    auto metadata = std::make_shared<TableMetaData>();
    metadata->sample = sample;
    return metadata;
}

bool TableMetaData::isPresent() const {
    return present;
}

// TODO Phase 2: Replace with real PostgreSQL catalog queries
// - Query pg_class for table row counts  
// - Query pg_attribute for column metadata
// - Query pg_stats for distinct value counts
// - Use INFORMATION_SCHEMA for type information