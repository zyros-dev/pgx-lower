#ifndef PGX_LOWER_RUNTIME_HEAP_H
#define PGX_LOWER_RUNTIME_HEAP_H

namespace pgx_lower::compiler::runtime {
class Heap {
public:
    Heap() = default;
    virtual ~Heap() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_HEAP_H
