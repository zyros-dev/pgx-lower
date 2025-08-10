#ifndef PGX_LOWER_RUNTIME_FLOATRUNTIME_H
#define PGX_LOWER_RUNTIME_FLOATRUNTIME_H
#include "runtime/helpers.h"
namespace pgx_lower::compiler::runtime {
struct FloatRuntime {
   static double sqrt(double);
   static double sin(double);
   static double cos(double);
   static double arcsin(double);
   static double exp(double);
   static double log(double);
   static double erf(double);
   static double pow(double, double);
};

} // namespace pgx_lower::compiler::runtime
#endif // PGX_LOWER_RUNTIME_FLOATRUNTIME_H
