#ifndef LINGODB_RUNTIME_SIMPLESTATE_H
#define LINGODB_RUNTIME_SIMPLESTATE_H
#include "runtime/ExecutionContext.h"
#include "runtime/ThreadLocal.h"

namespace pgx_lower::compiler::runtime {
class SimpleState {
   public:
   static uint8_t* create(size_t sizeOfType);
   static uint8_t* merge(ThreadLocal* threadLocal, void (*merge)(uint8_t* dest, uint8_t* src));
};
} //end namespace pgx_lower::compiler::runtime
#endif //LINGODB_RUNTIME_SIMPLESTATE_H
