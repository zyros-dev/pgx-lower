#ifndef RUNTIME_METADATA_H
#define RUNTIME_METADATA_H
#include <optional>
#include <unordered_map>
#include<variant>

// Stubbed PostgreSQL includes - no Arrow dependency
#include <string>
#include <vector>
#include <memory>
namespace runtime {
struct ColumnType {
   std::string base;
   bool nullable;
   std::vector<std::variant<size_t, std::string>> modifiers;
};
class ColumnMetaData {
   std::optional<size_t> distinctValues;
   ColumnType columnType;

   public:
   const std::optional<size_t>& getDistinctValues() const;
   void setDistinctValues(const std::optional<size_t>& distinctValues);
   const ColumnType& getColumnType() const;
   void setColumnType(const ColumnType& columnType);
};
class TableMetaData {
   bool present;
   size_t numRows;
   std::vector<std::string> primaryKey;
   std::unordered_map<std::string, std::shared_ptr<ColumnMetaData>> columns;
   std::vector<std::string> orderedColumns;
   // Stub: PostgreSQL result set (was arrow::RecordBatch)
   void* sample;  // Placeholder for PostgreSQL SPI result

   public:
   TableMetaData() : present(false) {}
   size_t getNumRows() const {
      return numRows;
   }
   void setNumRows(size_t numRows) {
      TableMetaData::numRows = numRows;
   }
   void addColumn(std::string name,std::shared_ptr<ColumnMetaData> columnMetaData){
      columns[name]=columnMetaData;
      orderedColumns.push_back(name);
   }
   void setPrimaryKey(const std::vector<std::string>& primaryKey) {
      TableMetaData::primaryKey = primaryKey;
   }
   const std::vector<std::string>& getPrimaryKey() const {
      return primaryKey;
   }
   const std::shared_ptr<ColumnMetaData> getColumnMetaData(const std::string& name) const {
      return columns.at(name);
   }
   void* getSample() const {
      return sample;  // Stub implementation
   }
   const std::vector<std::string>& getOrderedColumns() const;
   static std::shared_ptr<TableMetaData> deserialize(std::string);
   std::string serialize(bool serializeSample=true) const;
   static std::shared_ptr<TableMetaData> create(const std::string& json, const std::string& name, void* sample);
   bool isPresent() const;
};
} // end namespace runtime

#endif // RUNTIME_METADATA_H
