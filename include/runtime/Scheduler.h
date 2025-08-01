#ifndef PGX_LOWER_SCHEDULER_H
#define PGX_LOWER_SCHEDULER_H

#include <cstddef>

namespace pgx_lower {
namespace scheduler {

// Stub implementation for scheduler
inline size_t getNumWorkers() {
    return 1; // Single-threaded for now
}

inline size_t currentWorkerId() {
    return 0; // Always worker 0 for now
}

} // namespace scheduler
} // namespace pgx_lower

#endif // PGX_LOWER_SCHEDULER_H