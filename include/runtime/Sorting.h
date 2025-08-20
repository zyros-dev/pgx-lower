#ifndef PGX_LOWER_RUNTIME_SORTING_H
#define PGX_LOWER_RUNTIME_SORTING_H
#include "runtime/GrowingBuffer.h"

namespace pgx_lower::compiler::runtime {

bool canParallelSort(const size_t valueSize);
Buffer parallelSort(FlexibleBuffer& values, bool (*compareFn)(uint8_t*, uint8_t*));

} // end namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_SORTING_H