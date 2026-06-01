#include "pgx-lower/runtime/NumericRuntime.h"

extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "utils/numeric.h"
#include "utils/fmgrprotos.h"
}

#include <cstring>
#include <string>

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

uint64_t NumericRuntime::pgx_numeric_hash(Datum value) {
    return static_cast<uint64_t>(DatumGetUInt32(DirectFunctionCall1(hash_numeric, value)));
}

Datum NumericRuntime::pgx_numeric_from_string(VarLen32 value) {
    std::string str(value.data(), value.getLen());
    return DirectFunctionCall3(numeric_in, CStringGetDatum(str.c_str()), ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1));
}

VarLen32 NumericRuntime::pgx_numeric_to_string(Datum value) {
    char* str = DatumGetCString(DirectFunctionCall1(numeric_out, value));
    return VarLen32(reinterpret_cast<uint8_t*>(str), std::strlen(str));
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
