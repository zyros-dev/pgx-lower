#ifndef PGX_LOWER_RUNTIME_NUMERICRUNTIME_H
#define PGX_LOWER_RUNTIME_NUMERICRUNTIME_H

#include <cstdint>

#ifndef POSTGRES_H
using Datum = unsigned long;
#endif

namespace runtime {

// PG-native NUMERIC arithmetic/compare. Each delegates to PostgreSQL's own
// numeric_* backend function via DirectFunctionCall, so results are exactly what
// stock PG produces (arbitrary precision, NaN/Inf, any scale). Declared as a
// struct of static methods in namespace `runtime` so the runtime-header-tool
// emits mlir::util::FunctionSpec bindings (rt::NumericRuntime::*) for the JIT
// lowering to call. Kept free of C++-only argument types (Datum/int32_t only,
// no templates/refs/exceptions) so it stays bitcode-injectable later — see the
// pgx-lower-bitcode-injection design.
struct NumericRuntime {
    static Datum pgx_numeric_add(Datum left, Datum right);
    static Datum pgx_numeric_sub(Datum left, Datum right);
    static Datum pgx_numeric_mul(Datum left, Datum right);
    // PG numeric_cmp semantics: <0, 0, >0. NaN sorts equal to NaN and greater
    // than all non-NaN; Inf/-Inf ordered as PG defines. Returned verbatim.
    static int32_t pgx_numeric_cmp(Datum left, Datum right);
};

} // namespace runtime

#endif // PGX_LOWER_RUNTIME_NUMERICRUNTIME_H
