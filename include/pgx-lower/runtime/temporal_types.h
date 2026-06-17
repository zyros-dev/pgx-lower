#pragma once

#include <cstdint>

namespace pgx_lower::runtime {
struct PgIntervalValue {
    int64_t time;
    int32_t day;
    int32_t month;
};
} // namespace pgx_lower::runtime
