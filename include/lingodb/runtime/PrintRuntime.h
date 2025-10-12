#ifndef RUNTIME_PRINTRUNTIME_H
#define RUNTIME_PRINTRUNTIME_H
#include <cstdint>
namespace runtime {
struct PrintRuntime {
   static void print(const char* txt);
   static void printVal(void* ptr, int32_t len);
   static void printPtr(void* ptr, int32_t offset, int32_t len);
};
} // namespace runtime
#endif // RUNTIME_PRINTRUNTIME_H
