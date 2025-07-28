#ifndef PGX_LOWER_RUNTIME_THREADLOCAL_H
#define PGX_LOWER_RUNTIME_THREADLOCAL_H

namespace pgx_lower::compiler::runtime {
class ThreadLocal {
public:
    ThreadLocal() = default;
    virtual ~ThreadLocal() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_THREADLOCAL_H
