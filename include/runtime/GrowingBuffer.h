#ifndef PGX_LOWER_RUNTIME_GROWINGBUFFER_H
#define PGX_LOWER_RUNTIME_GROWINGBUFFER_H

namespace pgx_lower::compiler::runtime {
class GrowingBuffer {
public:
    GrowingBuffer() = default;
    virtual ~GrowingBuffer() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_GROWINGBUFFER_H
