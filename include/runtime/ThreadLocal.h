#ifndef PGX_LOWER_THREADLOCAL_H
#define PGX_LOWER_THREADLOCAL_H

#include <cstdint>

namespace pgx_lower::compiler::runtime {

// Minimal ThreadLocal stub for ArrowTable.h
class ThreadLocal {
public:
    uint8_t* getLocal() { return nullptr; }
    static ThreadLocal* create(uint8_t* (*initFn)(uint8_t*), uint8_t*) { return nullptr; }
};

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_THREADLOCAL_H