#ifndef PGX_LOWER_RUNTIME_SIMPLESTATE_H
#define PGX_LOWER_RUNTIME_SIMPLESTATE_H

namespace pgx_lower::compiler::runtime {
class SimpleState {
public:
    SimpleState() = default;
    virtual ~SimpleState() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_SIMPLESTATE_H
