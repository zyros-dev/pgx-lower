#include "lingodb/runtime/Vector.h"
#include <algorithm>
#include <csignal>
#include <csetjmp>
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "utils/memutils.h"
}

static bool isMemoryValid(void* ptr, size_t size) {
   if (!ptr) return false;
   
   uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
   if (addr < 0x1000 || addr > 0x7fffffffffff) {
      return false;
   }
   
   if (CurrentMemoryContext) {
      PGX_LOG(RUNTIME, TRACE, "CurrentMemoryContext is set: %p", CurrentMemoryContext);
   }
   
   return true;
}
runtime::Vector* runtime::Vector::create(size_t sizeOfType, size_t initialCapacity) {
   return new Vector(initialCapacity, sizeOfType);
}
void runtime::Vector::resize() {
   size_t newCapacity = cap * 2;
   ptr = runtime::MemoryHelper::resize(ptr, len * typeSize, newCapacity * typeSize);
   cap = newCapacity;
}
size_t runtime::Vector::getLen() const {
   PGX_LOG(RUNTIME, DEBUG, "Vector::getLen() called - this=%p, len=%zu, cap=%zu, ptr=%p", this, len, cap, ptr);
   return len;
}
size_t runtime::Vector::getCap() const {
   return cap;
}
uint8_t* runtime::Vector::getPtr() const {
   return ptr;
}
size_t runtime::Vector::getTypeSize() const {
   return typeSize;
}
void runtime::Vector::sort(bool (*compareFn)(uint8_t*, uint8_t*)) {
   PGX_IO(RUNTIME);

   if (!compareFn) {
      PGX_LOG(RUNTIME, DEBUG, "Vector::sort: compareFn is NULL, aborting sort");
      return;
   }
   
   if (len == 0) {
      PGX_LOG(RUNTIME, DEBUG, "Vector::sort: empty vector, nothing to sort");
      return;
   }
   
   PGX_LOG(RUNTIME, DEBUG, "Vector::sort: building pointer array for %zu elements", len);
   std::vector<uint8_t*> toSort;
   for (size_t i = 0; i < len; i++) {
      uint8_t* elemPtr = ptrAt<uint8_t>(i);
      
      if (i < 5) {
         PGX_LOG(RUNTIME, DEBUG, "Vector::sort: element[%zu] at %p (typeSize=%zu)", i, elemPtr, typeSize);
         
         if (typeSize > 0 && isMemoryValid(elemPtr, typeSize)) {
            std::string hexDump;
            for (size_t j = 0; j < std::min(typeSize, size_t(32)); j++) {
               char buf[4];
               snprintf(buf, sizeof(buf), "%02x ", elemPtr[j]);
               hexDump += buf;
            }
            PGX_LOG(RUNTIME, TRACE, "Vector::sort: element[%zu] hex: %s", i, hexDump.c_str());
         }
      } else {
         PGX_LOG(RUNTIME, TRACE, "Vector::sort: element[%zu] at %p", i, elemPtr);
      }
      
      toSort.push_back(elemPtr);
   }
   
   PGX_LOG(RUNTIME, DEBUG, "Vector::sort: calling std::sort with compareFn");
   
   size_t comparisons = 0;
  PGX_LOG(RUNTIME, TRACE, "Sorting...");
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      comparisons++;
      PGX_LOG(RUNTIME, TRACE, "Vector::sort: comparison #%zu: left=%p, right=%p", comparisons, left, right);
      
      if (comparisons <= 5) {
         volatile uint8_t testL = 0, testR = 0;
         try {
            testL = *left;
            testR = *right;
            PGX_LOG(RUNTIME, TRACE, "Vector::sort: pointers appear valid (first bytes: L=0x%02x, R=0x%02x)", testL, testR);
         } catch (...) {
            PGX_LOG(RUNTIME, DEBUG, "Vector::sort: WARNING - pointers may be invalid!");
         }
      }

      PGX_LOG(RUNTIME, TRACE, "Calling compare function now!");
      bool result = compareFn(left, right);
      PGX_LOG(RUNTIME, TRACE, "Vector::sort: compare(%p, %p) = %s", left, right, result ? "true" : "false");
      return result;
   });
   
   PGX_LOG(RUNTIME, DEBUG, "Vector::sort: allocating sorted buffer of %zu bytes", typeSize * len);
   uint8_t* sorted = new uint8_t[typeSize * len];
   
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = sorted + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   
   PGX_LOG(RUNTIME, DEBUG, "Vector::sort: copying sorted data back to vector");
   memcpy(ptr, sorted, typeSize * len);
   delete[] sorted;
   
   PGX_LOG(RUNTIME, IO, "Vector::sort OUT: success");
}
void runtime::Vector::destroy(Vector* vec) {
   delete vec;
}
