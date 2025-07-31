#ifndef PGX_LOWER_RUNTIME_HASHTABLE_H
#define PGX_LOWER_RUNTIME_HASHTABLE_H
#include "ThreadLocal.h"
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/helpers.h"
namespace pgx_lower::compiler::runtime {
class Hashtable {
   struct Entry {
      Entry* next;
      size_t hashValue;
      uint8_t content[];
      //kv follows
   };
   runtime::LegacyFixedSizedBuffer<Entry*> ht;
   size_t hashMask;
   runtime::FlexibleBuffer values;
   size_t typeSize;
   //initial value follows...
   Hashtable(size_t initialCapacity, size_t typeSize) : ht(initialCapacity * 2), hashMask(initialCapacity * 2 - 1), values(initialCapacity, typeSize), typeSize(typeSize) {}

   public:
   void resize();
   Entry* insert(size_t hash);
   static void lock(Entry* entry, size_t subtract);
   static void unlock(Entry* entry, size_t subtract);
   static Hashtable* create(size_t typeSize, size_t initialCapacity);
   static void destroy(Hashtable*);
   void mergeEntries(bool (*isEq)(uint8_t*, uint8_t*), void (*merge)(uint8_t*, uint8_t*), Hashtable* other);
   static runtime::Hashtable* merge(ThreadLocal*, bool (*eq)(uint8_t*, uint8_t*), void (*combine)(uint8_t*, uint8_t*));

   runtime::BufferIterator* createIterator();
};

} // end namespace pgx_lower::compiler::runtime
#endif // PGX_LOWER_RUNTIME_HASHTABLE_H
