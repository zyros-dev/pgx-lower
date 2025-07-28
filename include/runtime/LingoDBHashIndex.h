#ifndef PGX_LOWER_RUNTIME_LINGODBHASHINDEX_H
#define PGX_LOWER_RUNTIME_LINGODBHASHINDEX_H
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>
#include "runtime/storage/Index.h"
#include "runtime/Buffer.h"
#include "catalog/LingoDBTableCatalogEntry.h"
#include "utility/Serializer.h"
namespace pgx_lower::compiler::runtime {
//todo: HashIndex maps hash to logical row id
//todo: we persist (hash, logical row id), we can even cluster by hash and store the required hashtable size
//todo: we can also create the hash index in parallel
class HashIndexIteration;
class HashIndexAccess;
class LingoDBTable;
class LingoDBHashIndex : public Index {
   struct Entry {
      size_t hash;
      Entry* next;
      size_t rowId;
   };

   Entry** ht = nullptr;
   int64_t mask;
   runtime::FlexibleBuffer buffer;
   std::string filename;
   std::string dbDir;
   bool persist;
   pgx_lower::compiler::catalog::LingoDBTableCatalogEntry* table = nullptr;
   LingoDBTable* tableStorage;
   std::vector<std::string> indexedColumns;
   bool loaded = false;
   //void build();
   //void computeHashes();
   void rawInsert(size_t startRowId, std::shared_ptr<arrow::Table> t);
   void rawBuild();

   public:
   virtual void setDBDir(std::string dbDir) {
      this->dbDir = dbDir;
   };
   LingoDBHashIndex(std::string filename, std::vector<std::string> indexedColumns) : buffer(16, sizeof(Entry)), filename(filename), indexedColumns(indexedColumns) {}
   void setTable(pgx_lower::compiler::catalog::LingoDBTableCatalogEntry* table);
   void flush();
   void ensureLoaded() override;
   void appendRows(size_t startRowId, std::shared_ptr<arrow::RecordBatch> table) override;
   void bulkInsert(size_t startRowId, std::shared_ptr<arrow::Table> newRows) override;
   void setPersist(bool value) {
      persist = value;
      if (persist) {
         flush();
      }
   }
   void serialize(lingodb::utility::Serializer& serializer) const;
   static std::unique_ptr<LingoDBHashIndex> deserialize(lingodb::utility::Serializer& deserializer);
   friend class HashIndexAccess;
   friend class HashIndexIteration;
   ~LingoDBHashIndex();
};
class HashIndexAccess {
   LingoDBHashIndex& hashIndex;
   std::vector<size_t> colIds;
   std::vector<HashIndexIteration> iteration;

   public:
   HashIndexAccess(LingoDBHashIndex& hashIndex, std::vector<std::string> cols);
   HashIndexIteration* lookup(size_t hash);
   friend class HashIndexIteration;
};
class HashIndexIteration {
   HashIndexAccess& access;
   size_t hash;
   LingoDBHashIndex::Entry* current;
   std::vector<const ArrayView*> arrayViewPtrs;

   public:
   HashIndexIteration(HashIndexAccess& access, size_t hash, LingoDBHashIndex::Entry* current);
   void reset(size_t hash, LingoDBHashIndex::Entry* current) {
      this->hash = hash;
      this->current = current;
   }
   bool hasNext();
   void consumeRecordBatch(pgx_lower::compiler::runtime::BatchView* batchView);
};

} //end namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_LINGODBHASHINDEX_H
