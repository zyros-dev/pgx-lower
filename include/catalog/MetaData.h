//
// Created by michael on 3/17/25.
// Adapted for pgx-lower - removed Arrow dependencies
//

#ifndef PGX_LOWER_CATALOG_METADATA_H
#define PGX_LOWER_CATALOG_METADATA_H

#include "llvm/ADT/Hashing.h"
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace pgx_lower::utility {
class Serializer;
class Deserializer;
} //end namespace pgx_lower::utility

namespace pgx_lower::catalog {

// Stub implementation for catalog metadata without Arrow dependencies
class Sample {
public:
   Sample() = default;
   Sample(const std::string& data) : sampleData(data) {}
   
   // Stub methods for compatibility
   void serialize(pgx_lower::utility::Serializer& serializer) const {}
   void deserialize(pgx_lower::utility::Deserializer& deserializer) {}
   
private:
   std::string sampleData;
};

class TableMetaData {
public:
   TableMetaData() = default;
   TableMetaData(const std::string& name) : tableName(name) {}
   
   // Stub methods for compatibility
   const std::string& getName() const { return tableName; }
   void serialize(pgx_lower::utility::Serializer& serializer) const {}
   void deserialize(pgx_lower::utility::Deserializer& deserializer) {}
   
private:
   std::string tableName;
};

class TableMetaDataProvider {
public:
   TableMetaDataProvider() = default;
   virtual ~TableMetaDataProvider() = default;
   
   // Stub methods for compatibility
   virtual std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) {
      return std::make_shared<TableMetaData>(name);
   }
};

} // namespace pgx_lower::catalog

#endif //PGX_LOWER_CATALOG_METADATA_H