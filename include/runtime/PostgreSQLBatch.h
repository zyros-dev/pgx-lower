#ifndef RUNTIME_POSTGRESQL_BATCH_H
#define RUNTIME_POSTGRESQL_BATCH_H

#include <vector>
#include <string>
#include <memory>
#include <variant>
#include <cstdint>

namespace runtime {
namespace postgresql {

// PostgreSQL-compatible data types
using ColumnValue = std::variant<
    int32_t, int64_t, double, std::string, bool,
    std::vector<uint8_t>  // For binary data
>;

// PostgreSQL column representation
class PostgreSQLColumn {
private:
    std::string name_;
    std::string pgType_;  // PostgreSQL type name (e.g., "integer", "text", "timestamp")
    std::vector<ColumnValue> data_;
    std::vector<bool> nulls_;  // Track null values

public:
    PostgreSQLColumn(const std::string& name, const std::string& pgType)
        : name_(name), pgType_(pgType) {}

    const std::string& getName() const { return name_; }
    const std::string& getPostgreSQLType() const { return pgType_; }
    
    size_t getLength() const { return data_.size(); }
    
    void appendValue(const ColumnValue& value, bool isNull = false) {
        data_.push_back(value);
        nulls_.push_back(isNull);
    }
    
    const ColumnValue& getValue(size_t index) const { return data_[index]; }
    bool isNull(size_t index) const { return nulls_[index]; }
    
    // PostgreSQL SPI integration placeholders
    void* getPostgreSQLDatum(size_t index) const { 
        // TODO: Convert to PostgreSQL Datum format
        return nullptr; 
    }
};

// PostgreSQL record batch equivalent
class PostgreSQLRecordBatch {
private:
    std::vector<std::shared_ptr<PostgreSQLColumn>> columns_;
    size_t numRows_;

public:
    PostgreSQLRecordBatch() : numRows_(0) {}
    
    void addColumn(std::shared_ptr<PostgreSQLColumn> column) {
        columns_.push_back(column);
        if (numRows_ == 0 && !columns_.empty()) {
            numRows_ = column->getLength();
        }
    }
    
    size_t getNumColumns() const { return columns_.size(); }
    size_t getNumRows() const { return numRows_; }
    
    std::shared_ptr<PostgreSQLColumn> getColumn(size_t index) const {
        return columns_[index];
    }
    
    std::shared_ptr<PostgreSQLColumn> getColumnByName(const std::string& name) const {
        for (auto& col : columns_) {
            if (col->getName() == name) {
                return col;
            }
        }
        return nullptr;
    }
    
    // PostgreSQL SPI result conversion
    void* toPostgreSQLResult() const {
        // TODO: Convert to PostgreSQL SPI result format
        return nullptr;
    }
};

} // namespace postgresql
} // namespace runtime

#endif // RUNTIME_POSTGRESQL_BATCH_H