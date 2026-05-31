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

int32_t NumericRuntime::pgx_numeric_cmp(Datum left, Datum right) {
    return DatumGetInt32(DirectFunctionCall2(numeric_cmp, left, right));
}

Datum NumericRuntime::pgx_i128_to_numeric(__int128 value, int32_t scale) {
    // Delegates to the existing scaled-i128 → Numeric builder. (That helper is
    // removed in PR3 once nothing else depends on the i128 path; this stub
    // remains as the scan-decode / constant materialization entry point.)
    return i128_to_numeric(value, scale);
}

} // namespace runtime
