#ifndef PGX_LOWER_RUNTIME_TIMING_H
#define PGX_LOWER_RUNTIME_TIMING_H
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
#endif //PGX_LOWER_RUNTIME_TIMING_H
