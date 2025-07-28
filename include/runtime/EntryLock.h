#ifndef PGX_LOWER_RUNTIME_ENTRYLOCK_H
#define PGX_LOWER_RUNTIME_ENTRYLOCK_H

namespace pgx_lower::compiler::runtime {
class EntryLock {
public:
    EntryLock() = default;
    virtual ~EntryLock() = default;
};
} // end namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_ENTRYLOCK_H
