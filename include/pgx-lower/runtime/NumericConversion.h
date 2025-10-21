#ifndef RUNTIME_NUMERICCONVERSION_H
#define RUNTIME_NUMERICCONVERSION_H

#include <cstdint>

#ifndef POSTGRES_H
using Datum = unsigned long;
#endif

extern "C" {

__int128 numeric_to_i128(Datum numeric_datum, int32_t target_scale);
Datum i128_to_numeric(__int128 value, int32_t scale);

} // extern "C"

#endif // RUNTIME_NUMERICCONVERSION_H
