#ifndef PGX_LOWER_RUNTIME_THREADLOCAL_H
#define PGX_LOWER_RUNTIME_THREADLOCAL_H
#include "scheduler/Scheduler.h"

#include <span>
namespace pgx_lower::compiler::runtime {
class ThreadLocal {
   uint8_t** values;
   uint8_t* (*initFn)(uint8_t*);
   uint8_t* arg;
   ThreadLocal(uint8_t* (*initFn)(uint8_t*), uint8_t* arg) : initFn(initFn), arg(arg) {
      values = new uint8_t*[pgx_lower::compiler::scheduler::getNumWorkers()];
      for (size_t i = 0; i < pgx_lower::compiler::scheduler::getNumWorkers(); i++) {
         values[i] = nullptr;
      }
   }

   public:
   uint8_t* getLocal();
   static ThreadLocal* create(uint8_t* (*initFn)(uint8_t*), uint8_t*);
   template <class T>
   std::span<T*> getThreadLocalValues() {
      for (size_t i = 0; i < pgx_lower::compiler::scheduler::getNumWorkers(); i++) {
         if (values[i]) break;
         if (i == pgx_lower::compiler::scheduler::getNumWorkers() - 1) {
            values[i] = initFn(arg);
         }
      }
      return std::span<T*>(reinterpret_cast<T**>(values), pgx_lower::compiler::scheduler::getNumWorkers());
   }
   uint8_t* merge(void (*mergeFn)(uint8_t*, uint8_t*));
   ~ThreadLocal() {
      delete[] values;
   }
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_THREADLOCAL_H
