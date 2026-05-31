#include "pgx-lower/runtime/NumericRuntime.h"
#include "pgx-lower/runtime/NumericConversion.h"

extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "utils/numeric.h"
#include "utils/fmgrprotos.h"
}

// Each method calls PG's own numeric_* via the fmgr direct-call ABI. Results are
// palloc'd in CurrentMemoryContext (the caller's per-tuple/per-query context).
namespace runtime {

Datum NumericRuntime::pgx_numeric_add(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_add, left, right);
}

Datum NumericRuntime::pgx_numeric_sub(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_sub, left, right);
}

Datum NumericRuntime::pgx_numeric_mul(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_mul, left, right);
}

Datum NumericRuntime::pgx_numeric_div(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_div, left, right);
}

Datum NumericRuntime::pgx_numeric_mod(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_mod, left, right);
}

int32_t NumericRuntime::pgx_numeric_cmp(Datum left, Datum right) {
    return DatumGetInt32(DirectFunctionCall2(numeric_cmp, left, right));
}

Datum NumericRuntime::pgx_i128_to_numeric(__int128 value, int32_t scale) {
    // Delegates to the existing scaled-i128 → Numeric builder. (That helper is
    // removed in PR3 once nothing else depends on the i128 path; this stub
    // remains as the scan-decode / constant materialization entry point.)
    return i128_to_numeric(value, scale);
}

__int128 NumericRuntime::pgx_numeric_to_i128(Datum numeric_datum, int32_t scale) {
    // Bridge back to the i128 columnar storage (PR2 only; removed in PR3).
    return numeric_to_i128(numeric_datum, scale);
}

Datum NumericRuntime::pgx_int_to_numeric(int64_t value) {
    return DirectFunctionCall1(int8_numeric, Int64GetDatum(value));
}

Datum NumericRuntime::pgx_float_to_numeric(double value) {
    return DirectFunctionCall1(float8_numeric, Float8GetDatum(value));
}

int64_t NumericRuntime::pgx_numeric_to_int(Datum numeric_datum) {
    return DatumGetInt64(DirectFunctionCall1(numeric_int8, numeric_datum));
}

double NumericRuntime::pgx_numeric_to_float(Datum numeric_datum) {
    return DatumGetFloat8(DirectFunctionCall1(numeric_float8, numeric_datum));
}

} // namespace runtime
