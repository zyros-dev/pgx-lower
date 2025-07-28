#ifndef PGX_LOWER_RUNTIME_HASHTABLE_H
#define PGX_LOWER_RUNTIME_HASHTABLE_H

namespace pgx_lower::compiler::runtime {
class Hashtable {
public:
    Hashtable() = default;
    virtual ~Hashtable() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_HASHTABLE_H
