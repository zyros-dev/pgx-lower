#ifndef PGX_LOWER_CATALOG_CATALOG_H
#define PGX_LOWER_CATALOG_CATALOG_H
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include "utility/Serialization.h"
namespace pgx_lower::compiler::catalog {
class Catalog;
class CatalogEntry {
   public:
   enum class CatalogEntryType : uint8_t {
      INVALID_ENTRY = 0,
      PGX_LOWER_TABLE_ENTRY = 1,
      PGX_LOWER_HASH_INDEX_ENTRY = 2,
   };

   protected:
   CatalogEntryType entryType;
   Catalog* catalog = nullptr;
   CatalogEntry(CatalogEntryType entryType) : entryType(entryType) {}

   public:
   virtual void setCatalog(Catalog* catalog) { this->catalog = catalog; }
   CatalogEntryType getEntryType() { return entryType; }
   virtual void setDBDir(std::string dbDir) {}
   virtual std::string getName() = 0;
   void serialize(pgx_lower::compiler::utility::Serializer& serializer) const;
   static std::shared_ptr<CatalogEntry> deserialize(pgx_lower::compiler::utility::Deserializer& deSerializer);
   virtual void serializeEntry(pgx_lower::compiler::utility::Serializer& serializer) const = 0;
   virtual void flush() {}
   virtual void setShouldPersist(bool shouldPersist) {}
   virtual void ensureFullyLoaded() {}
   virtual ~CatalogEntry() = default;
};

class Catalog {
   static constexpr size_t binaryVersion = 2;
   bool shouldPersist;
   std::string dbDir;

   public:
   Catalog() : shouldPersist(false) {}
   std::string getDbDir() const { return dbDir; }
   void serialize(pgx_lower::compiler::utility::Serializer& serializer) const;
   static Catalog deserialize(pgx_lower::compiler::utility::Deserializer& deSerializer);

   std::optional<std::shared_ptr<CatalogEntry>> getEntry(std::string name) {
      if (entries.contains(name)) {
         return entries.at(name);
      } else {
         return std::nullopt;
      }
   }
   template <class T>
   std::optional<std::shared_ptr<T>> getTypedEntry(std::string name) {
      if (entries.contains(name)) {
         auto entry = entries.at(name);
         for (auto x : T::entryTypes) {
            if (entry->getEntryType() == x) {
               return std::static_pointer_cast<T>(entry);
            }
         }
         return std::nullopt;
      } else {
         return std::nullopt;
      }
   }
   void persist();
   void setShouldPersist(bool shouldPersist) {
      this->shouldPersist = shouldPersist;
      for (auto& entry : entries) {
         entry.second->setShouldPersist(shouldPersist);
      }
   }

   void insertEntry(std::shared_ptr<CatalogEntry> entry);
   static std::shared_ptr<Catalog> create(std::string dbDir, bool eagerLoading);
   static std::shared_ptr<Catalog> createEmpty();
   ~Catalog() {
      persist();
   }

   private:
   std::unordered_map<std::string, std::shared_ptr<CatalogEntry>> entries;
};

} // namespace pgx_lower::compiler::catalog

#endif //PGX_LOWER_CATALOG_CATALOG_H
