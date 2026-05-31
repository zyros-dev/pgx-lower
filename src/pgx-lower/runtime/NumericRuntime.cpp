#include "pgx-lower/runtime/NumericRuntime.h"

extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "utils/numeric.h"
#include "utils/fmgrprotos.h"
}

// Each stub calls PG's own numeric_* via the fmgr direct-call ABI. Results are
// palloc'd in CurrentMemoryContext (the caller's per-tuple/per-query context).
Datum pgx_numeric_add(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_add, left, right);
}

Datum pgx_numeric_sub(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_sub, left, right);
}

Datum pgx_numeric_mul(Datum left, Datum right) {
    return DirectFunctionCall2(numeric_mul, left, right);
}

int32_t pgx_numeric_cmp(Datum left, Datum right) {
    return DatumGetInt32(DirectFunctionCall2(numeric_cmp, left, right));
}
