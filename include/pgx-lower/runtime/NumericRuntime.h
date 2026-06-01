#ifndef PGX_LOWER_RUNTIME_NUMERICRUNTIME_H
#define PGX_LOWER_RUNTIME_NUMERICRUNTIME_H

#include <cstdint>

#include "lingodb/runtime/helpers.h"

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
    static Datum pgx_numeric_div(Datum left, Datum right);
    static Datum pgx_numeric_mod(Datum left, Datum right);
    // PG numeric_cmp semantics: <0, 0, >0. NaN sorts equal to NaN and greater
    // than all non-NaN; Inf/-Inf ordered as PG defines. Returned verbatim.
    static int32_t pgx_numeric_cmp(Datum left, Datum right);
    static uint64_t pgx_numeric_hash(Datum value);
    static Datum pgx_numeric_from_string(VarLen32 value);
    static VarLen32 pgx_numeric_to_string(Datum value);

    // Cast bridges (PG-native). Used by the DB->Std cast lowering now that a
    // decimal is a Numeric datum rather than an i128.
    static Datum pgx_int_to_numeric(int64_t value);   // int8_numeric
    static Datum pgx_float_to_numeric(double value);  // float8_numeric
    static int64_t pgx_numeric_to_int(Datum numeric_datum);  // numeric_int8
    static double pgx_numeric_to_float(Datum numeric_datum); // numeric_float8
};

} // namespace runtime

#endif // PGX_LOWER_RUNTIME_NUMERICRUNTIME_H
