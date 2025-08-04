#ifndef PGX_LOWER_RUNTIME_ENTRYLOCK_H
#define PGX_LOWER_RUNTIME_ENTRYLOCK_H
#include <atomic>

namespace pgx_lower::compiler::runtime {
class EntryLock {
   std::atomic_flag m{};

   public:
   static void lock(EntryLock* lock);
   static void unlock(EntryLock* lock);
   static void initialize(EntryLock* lock);
};
static_assert(sizeof(EntryLock) <= 8, "SpinLock is too big");

} // namespace pgx_lower::compiler::runtime

#endif //PGX_LOWER_RUNTIME_ENTRYLOCK_H
