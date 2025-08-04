#ifndef LINGODB_RUNTIME_SORTING_H
#define LINGODB_RUNTIME_SORTING_H
#include "runtime/GrowingBuffer.h"

namespace pgx_lower::compiler::runtime {

bool canParallelSort(const size_t valueSize);
Buffer parallelSort(FlexibleBuffer& values, bool (*compareFn)(uint8_t*, uint8_t*));

} // end namespace pgx_lower::compiler::runtime
#endif //LINGODB_RUNTIME_SORTING_H