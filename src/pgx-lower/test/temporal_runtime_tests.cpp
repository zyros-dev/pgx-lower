extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
#include "utils/timestamp.h"
}

#include "lingodb/runtime/DateRuntime.h"
#include "pgx-lower/runtime/runtime_templates.h"
#include "pgx-lower/test/pgx_test_fn.h"

#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

extern "C" bool pgx_lower_runtime_type_oid_supported_for_testing(Oid typeOid);
extern "C" const char* pgx_lower_runtime_column_type_name_for_testing(Oid typeOid);
extern "C" const char* pgx_lower_runtime_unsupported_message_for_testing(Oid typeOid);

PGX_TEST_FN(date_extract_year_pg_epoch) {
    const DateADT pgEpoch = 0;

    REQUIRE(runtime::DateRuntime::extractYear(pgEpoch) == 2000);
    PG_RETURN_VOID();
}

PGX_TEST_FN(date_extract_year_before_epoch) {
    const DateADT oneYearBeforePgEpoch = -365;

    REQUIRE(runtime::DateRuntime::extractYear(oneYearBeforePgEpoch) == 1999);
    PG_RETURN_VOID();
}

PGX_TEST_FN(date_extract_month_day_boundary) {
    const DateADT january31_2000 = 30;

    REQUIRE(runtime::DateRuntime::extractMonth(january31_2000) == 1);
    REQUIRE(runtime::DateRuntime::extractDay(january31_2000) == 31);
    PG_RETURN_VOID();
}

PGX_TEST_FN(temporal_interval_from_datum_preserves_fields) {
    Interval interval{};
    interval.time = 123456789;
    interval.day = -7;
    interval.month = 14;

    const auto value = pgx_lower::runtime::fromDatum<pgx_lower::runtime::PgIntervalValue>(IntervalPGetDatum(&interval),
                                                                                          INTERVALOID);

    REQUIRE(value.time == 123456789);
    REQUIRE(value.day == -7);
    REQUIRE(value.month == 14);
    PG_RETURN_VOID();
}

PGX_TEST_FN(temporal_interval_to_datum_preserves_fields) {
    const pgx_lower::runtime::PgIntervalValue value{123456789, -7, 14};

    const Datum datum = pgx_lower::runtime::toDatum<pgx_lower::runtime::PgIntervalValue>(value);
    const auto* interval = DatumGetIntervalP(datum);

    REQUIRE(interval->time == 123456789);
    REQUIRE(interval->day == -7);
    REQUIRE(interval->month == 14);
    PG_RETURN_VOID();
}

PGX_TEST_FN(temporal_interval_datum_roundtrip_preserves_fields) {
    Interval source{};
    source.time = 123456789;
    source.day = -7;
    source.month = 14;

    const auto value = pgx_lower::runtime::fromDatum<pgx_lower::runtime::PgIntervalValue>(IntervalPGetDatum(&source),
                                                                                          INTERVALOID);
    const Datum roundtripped = pgx_lower::runtime::toDatum<pgx_lower::runtime::PgIntervalValue>(value);
    const auto* interval = DatumGetIntervalP(roundtripped);

    REQUIRE(interval->time == 123456789);
    REQUIRE(interval->day == -7);
    REQUIRE(interval->month == 14);
    PG_RETURN_VOID();
}

PGX_TEST_FN(temporal_runtime_rejects_timestamptz_oid) {
    REQUIRE(!pgx_lower_runtime_type_oid_supported_for_testing(TIMESTAMPTZOID));
    REQUIRE(pgx_lower_runtime_column_type_name_for_testing(TIMESTAMPTZOID) == nullptr);
    const char* message = pgx_lower_runtime_unsupported_message_for_testing(TIMESTAMPTZOID);
    REQUIRE(message != nullptr);
    REQUIRE(std::string(message)
            == "unsupported temporal type TIMESTAMPTZOID; timezone semantics are not supported by pgx-lower");
    PG_RETURN_VOID();
}
