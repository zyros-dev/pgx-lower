extern "C" {
#include "postgres.h"
#include "access/tupdesc.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "nodes/pg_list.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "varatt.h"
#include "utils/numeric.h"
#include "utils/fmgrprotos.h"
}

#include <cstring>
#include <vector>

#include "pgx-lower/execution/postgres/my_executor.h"
#include "pgx-lower/runtime/NumericRuntime.h"
#include "pgx-lower/test/pgx_test_fn.h"

TupleDesc setupTupleDescriptor(const PlannedStmt* stmt, const std::vector<int>& selectedColumns);

namespace {

#define NBASE 10000
#define NUMERIC_POS 0x0000
#define NUMERIC_NEG 0x4000
#define NUMERIC_DSCALE_MASK 0x3FFF

using NumericDigit = int16;

struct NumericLong {
    uint16 n_sign_dscale;
    int16 n_weight{};
    NumericDigit n_data[1];
};

union NumericChoice {
    uint16 n_header{};
    struct NumericLong n_long;
};

struct NumericData {
    int32 vl_len_{};
    union NumericChoice choice{};
};

Datum make_numeric(std::vector<char>& buf, bool neg, int16 weight, uint16 dscale,
                   const std::vector<NumericDigit>& digits) {
    const size_t hdr = VARHDRSZ + sizeof(uint16) + sizeof(int16);
    const size_t total = hdr + digits.size() * sizeof(NumericDigit);
    buf.assign(total, 0);
    auto* num = reinterpret_cast<NumericData*>(buf.data());
    SET_VARSIZE(num, total);
    num->choice.n_long.n_sign_dscale = (neg ? NUMERIC_NEG : NUMERIC_POS) | (dscale & NUMERIC_DSCALE_MASK);
    num->choice.n_long.n_weight = weight;
    std::memcpy(num->choice.n_long.n_data, digits.data(), digits.size() * sizeof(NumericDigit));
    return PointerGetDatum(num);
}

Const make_result_const(Oid typeOid, int32_t typmod, Oid collation = InvalidOid) {
    Const c{};
    c.xpr.type = T_Const;
    c.consttype = typeOid;
    c.consttypmod = typmod;
    c.constcollid = collation;
    c.constisnull = false;
    c.constbyval = false;
    c.constlen = -1;
    c.constvalue = Datum{0};
    return c;
}

TupleDesc build_result_descriptor(Expr* expr) {
    TargetEntry target{};
    target.xpr.type = T_TargetEntry;
    target.expr = expr;
    target.resno = 1;
    target.resname = const_cast<char*>("result");
    target.resjunk = false;

    Plan plan{};
    plan.type = T_Result;
    plan.targetlist = list_make1(&target);

    PlannedStmt stmt{};
    stmt.type = T_PlannedStmt;
    stmt.planTree = &plan;

    return setupTupleDescriptor(&stmt, {-1});
}

}  // namespace

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        auto _a = static_cast<uint32_t>(actual);                                                                       \
        auto _e = static_cast<uint32_t>(expected);                                                                     \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        auto _a = static_cast<int32_t>(actual);                                                                        \
        auto _e = static_cast<int32_t>(expected);                                                                      \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define ASSERT_NUMERIC_EQ_STR(actual_datum, expected_cstr)                                                             \
    do {                                                                                                               \
        char* _s = DatumGetCString(DirectFunctionCall1(numeric_out, (actual_datum)));                                  \
        if (std::strcmp(_s, (expected_cstr)) != 0)                                                                     \
            elog(ERROR, "%s:%d expected numeric '%s' got '%s'", __FILE__, __LINE__, (expected_cstr), _s);              \
    } while (0)

PGX_TEST_FN(numeric_result_descriptor_preserves_typmod) {
    auto c = make_result_const(NUMERICOID, 786438);
    const TupleDesc desc = build_result_descriptor(reinterpret_cast<Expr*>(&c));
    const Form_pg_attribute attr = TupleDescAttr(desc, 0);
    REQUIRE_EQ_U32(attr->atttypid, NUMERICOID);
    REQUIRE_EQ_I32(attr->atttypmod, 786438);
    FreeTupleDesc(desc);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_result_descriptor_preserves_string_collation) {
    auto c = make_result_const(VARCHAROID, 18, DEFAULT_COLLATION_OID);
    const TupleDesc desc = build_result_descriptor(reinterpret_cast<Expr*>(&c));
    const Form_pg_attribute attr = TupleDescAttr(desc, 0);
    REQUIRE_EQ_U32(attr->atttypid, VARCHAROID);
    REQUIRE_EQ_I32(attr->atttypmod, 18);
    REQUIRE_EQ_U32(attr->attcollation, DEFAULT_COLLATION_OID);
    FreeTupleDesc(desc);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_add_basic) {
    std::vector<char> buf_l;
    std::vector<char> buf_r;
    Datum l = make_numeric(buf_l, false, 1, 0, {1, 2345}); // 12345
    Datum r = make_numeric(buf_r, false, 0, 0, {6789}); // 6789
    ASSERT_NUMERIC_EQ_STR(runtime::NumericRuntime::pgx_numeric_add(l, r), "19134");
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_sub_basic) {
    std::vector<char> buf_l;
    std::vector<char> buf_r;
    Datum l = make_numeric(buf_l, false, 1, 0, {1, 2345}); // 12345
    Datum r = make_numeric(buf_r, false, 0, 0, {6789}); // 6789
    ASSERT_NUMERIC_EQ_STR(runtime::NumericRuntime::pgx_numeric_sub(l, r), "5556");
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_mul_basic) {
    std::vector<char> buf_l;
    std::vector<char> buf_r;
    Datum l = make_numeric(buf_l, false, 0, 0, {123}); // 123
    Datum r = make_numeric(buf_r, false, 0, 0, {456}); // 456
    ASSERT_NUMERIC_EQ_STR(runtime::NumericRuntime::pgx_numeric_mul(l, r), "56088");
    PG_RETURN_VOID();
}

#define ASSERT_CMP_SIGN(actual, expected_sign)                                                                         \
    do {                                                                                                               \
        int32_t _a = (actual);                                                                                         \
        int32_t _e = (expected_sign);                                                                                  \
        int _as = (_a > 0) - (_a < 0);                                                                                 \
        if (_as != _e)                                                                                                 \
            elog(ERROR, "%s:%d expected cmp sign %d got %d (raw %d)", __FILE__, __LINE__, _e, _as, _a);                \
    } while (0)

PGX_TEST_FN(numeric_cmp_lt) {
    std::vector<char> a;
    std::vector<char> b;
    ASSERT_CMP_SIGN(runtime::NumericRuntime::pgx_numeric_cmp(make_numeric(a, false, 0, 0, {100}),
                                                             make_numeric(b, false, 0, 0, {200})),
                    -1);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_cmp_eq) {
    std::vector<char> a;
    std::vector<char> b;
    ASSERT_CMP_SIGN(runtime::NumericRuntime::pgx_numeric_cmp(make_numeric(a, false, 0, 0, {4242}),
                                                             make_numeric(b, false, 0, 0, {4242})),
                    0);
    PG_RETURN_VOID();
}

// 40-digit value: out of __int128 range. Build via numeric_in (string parse)
// rather than make_numeric so we don't hand-pack NBASE digits.
PGX_TEST_FN(numeric_cmp_wide) {
    Datum big = DirectFunctionCall3(numeric_in, CStringGetDatum("12345678901234567890123456789012345678901"),
                                    ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1));
    Datum small = DirectFunctionCall3(numeric_in, CStringGetDatum("1"), ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1));
    ASSERT_CMP_SIGN(runtime::NumericRuntime::pgx_numeric_cmp(big, small), 1);
    PG_RETURN_VOID();
}

PGX_TEST_FN(numeric_cmp_nan_gt_finite) {
    Datum nan = DirectFunctionCall3(numeric_in, CStringGetDatum("NaN"), ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1));
    Datum finite = DirectFunctionCall3(numeric_in, CStringGetDatum("999999"), ObjectIdGetDatum(InvalidOid),
                                       Int32GetDatum(-1));
    ASSERT_CMP_SIGN(runtime::NumericRuntime::pgx_numeric_cmp(nan, finite), 1);
    PG_RETURN_VOID();
}
