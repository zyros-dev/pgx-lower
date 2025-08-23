#ifndef PGX_LOWER_RUNTIME_DECIMALRUNTIME_H
#define PGX_LOWER_RUNTIME_DECIMALRUNTIME_H
#include "lingodb/runtime/helpers.h"
namespace pgx_lower::compiler::runtime {
struct DecimalRuntime {
   static __int128 round(__int128 value, int64_t digits, int64_t scale);
};
} // namespace pgx_lower::compiler::runtime
#endif // PGX_LOWER_RUNTIME_DECIMALRUNTIME_H
