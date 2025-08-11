/**
 * Stub implementation of MetaData.cpp - PostgreSQL conversion
 * Removes Arrow/JSON dependencies while keeping same interface
 */

#include "runtime/metadata.h"

namespace runtime {

// ColumnMetaData stub implementations
const std::optional<size_t>& ColumnMetaData::getDistinctValues() const {
    return distinctValues;
}

void ColumnMetaData::setDistinctValues(const std::optional<size_t>& distinctValues) {
    ColumnMetaData::distinctValues = distinctValues;
}

const ColumnType& ColumnMetaData::getColumnType() const {
    return columnType;
}

void ColumnMetaData::setColumnType(const ColumnType& columnType) {
    ColumnMetaData::columnType = columnType;
}

// TableMetaData stub implementations
const std::vector<std::string>& TableMetaData::getOrderedColumns() const {
    return orderedColumns;
}

std::shared_ptr<TableMetaData> TableMetaData::deserialize(std::string json) {
    // Stub: Return empty metadata (no JSON parsing)
    return std::make_shared<TableMetaData>();
}

std::string TableMetaData::serialize(bool serializeSample) const {
    // Stub: Return empty JSON string
    return "{}";
}

std::shared_ptr<TableMetaData> TableMetaData::create(const std::string& json, const std::string& name, void* sample) {
    // Stub: Create basic metadata without Arrow dependencies
    auto res = std::make_shared<TableMetaData>();
    res->present = true;
    res->numRows = 0;  // Placeholder
    res->sample = sample;
    return res;
}

bool TableMetaData::isPresent() const {
    return present;
}

} // end namespace runtime