#ifndef LINGODB_RUNTIME_DATERUNTIME_H
#define LINGODB_RUNTIME_DATERUNTIME_H
#include "runtime/helpers.h"
namespace pgx_lower::compiler::runtime {
struct DateRuntime {
   static int64_t extractHour(int64_t date);
   static int64_t extractMinute(int64_t date);
   static int64_t extractSecond(int64_t date);
   static int64_t extractDay(int64_t date);
   static int64_t extractMonth(int64_t date);
   static int64_t extractYear(int64_t date);
   static int64_t subtractMonths(int64_t date, int64_t months);
   static int64_t addMonths(int64_t date, int64_t months);
   static int64_t dateDiffSeconds(int64_t start, int64_t end);
   static int64_t dateTrunc(VarLen32 part, int64_t date);
};
} // namespace pgx_lower::compiler::runtime
#endif // LINGODB_RUNTIME_DATERUNTIME_H
