#ifndef LINGODB_RUNTIME_TIMING_H
#define LINGODB_RUNTIME_TIMING_H
#include <cstdint>
namespace pgx_lower::compiler::runtime {
class Timing {
   public:
   static uint64_t start();
   static void startPerf();
   static void stopPerf();
   static void stop(uint64_t start);
};
} // end namespace pgx_lower::compiler::runtime
#endif //LINGODB_RUNTIME_TIMING_H
