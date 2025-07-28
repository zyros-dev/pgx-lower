#ifndef PGX_LOWER_RUNTIME_TRACING_H
#define PGX_LOWER_RUNTIME_TRACING_H

namespace pgx_lower::compiler::runtime {
class Tracing {
public:
    Tracing() = default;
    virtual ~Tracing() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_TRACING_H
