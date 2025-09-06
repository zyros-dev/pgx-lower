#ifndef RUNTIME_STRINGRUNTIME_H
#define RUNTIME_STRINGRUNTIME_H
#include "lingodb/runtime/helpers.h"
namespace runtime {
struct StringRuntime {
   static bool NO_SIDE_EFFECTS compareEq(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS compareNEq(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS compareLt(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS compareGt(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS compareLte(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS compareGte(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS like(VarLen32 l, VarLen32 r);
   static bool NO_SIDE_EFFECTS startsWith(VarLen32 str, VarLen32 substr);
   static bool NO_SIDE_EFFECTS endsWith(VarLen32 str, VarLen32 substr);
   static NO_SIDE_EFFECTS int64_t toInt(VarLen32 str);
   static NO_SIDE_EFFECTS float toFloat32(VarLen32 str);
   static NO_SIDE_EFFECTS double toFloat64(VarLen32 str);
   static NO_SIDE_EFFECTS __int128 toDecimal(VarLen32 str, int32_t reqScale);
   static VarLen32 fromInt(int64_t);
   static VarLen32 fromFloat32(float);
   static VarLen32 fromFloat64(double);
   static VarLen32 fromChar(uint64_t, size_t bytes);
   static VarLen32 fromDecimal(__int128, int32_t scale);
   static VarLen32 substr(VarLen32 str, size_t from,size_t to);
   static NO_SIDE_EFFECTS size_t findMatch(VarLen32 str,VarLen32 needle, size_t start, size_t end);
   
   // String concatenation operations
   static VarLen32 concat(VarLen32 left, VarLen32 right);
   static VarLen32 concat3(VarLen32 a, VarLen32 b, VarLen32 c);
   
   // Case conversion operations
   static VarLen32 upper(VarLen32 str);
   static VarLen32 lower(VarLen32 str);
   
   // PostgreSQL-style SUBSTRING (FROM pos FOR length)
   static VarLen32 substring(VarLen32 str, int32_t start, int32_t length);
   
   // Length operations
   static NO_SIDE_EFFECTS int32_t length(VarLen32 str);
   static NO_SIDE_EFFECTS int32_t charLength(VarLen32 str);
   
   // Trimming operations
   static VarLen32 trim(VarLen32 str);
   static VarLen32 ltrim(VarLen32 str);
   static VarLen32 rtrim(VarLen32 str);
   
   // Case-insensitive pattern matching
   static bool NO_SIDE_EFFECTS ilike(VarLen32 str, VarLen32 pattern);
};
} // namespace runtime
#endif // RUNTIME_STRINGRUNTIME_H
