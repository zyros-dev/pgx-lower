#ifndef PGX_LOWER_RUNTIME_LAZYJOINHASHTABLE_H
#define PGX_LOWER_RUNTIME_LAZYJOINHASHTABLE_H

namespace pgx_lower::compiler::runtime {
class LazyJoinHashtable {
public:
    LazyJoinHashtable() = default;
    virtual ~LazyJoinHashtable() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_LAZYJOINHASHTABLE_H
