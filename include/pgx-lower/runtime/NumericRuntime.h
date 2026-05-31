#ifndef PGX_LOWER_RUNTIME_NUMERICRUNTIME_H
#define PGX_LOWER_RUNTIME_NUMERICRUNTIME_H

#include <cstdint>

#ifndef POSTGRES_H
using Datum = unsigned long;
#endif

// Clean-C NUMERIC arithmetic/compare stubs. Each delegates to PostgreSQL's own
// numeric_* backend function via DirectFunctionCall, so results are exactly what
// stock PG produces (arbitrary precision, NaN/Inf, any scale). Kept free of
// C++-only constructs (no templates/refs/exceptions) so they can later be
// emitted as bitcode and inlined into the JIT'd query — see the
// pgx-lower-bitcode-injection design. Do not add C++ features here.
extern "C" {

Datum pgx_numeric_add(Datum left, Datum right);
Datum pgx_numeric_sub(Datum left, Datum right);
Datum pgx_numeric_mul(Datum left, Datum right);
// PG numeric_cmp semantics: <0, 0, >0. NaN sorts equal to NaN and greater than
// all non-NaN; Inf/-Inf ordered as PG defines. We return PG's value verbatim.
int32_t pgx_numeric_cmp(Datum left, Datum right);

} // extern "C"

#endif // PGX_LOWER_RUNTIME_NUMERICRUNTIME_H
