#include "lingodb/runtime/DateRuntime.h"
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "utils/date.h"
#include "utils/timestamp.h"
#include "utils/datetime.h"
#include "datatype/timestamp.h"
}

// Convert nanoseconds since PostgreSQL epoch (2000-01-01) to PostgreSQL DateADT (days since 2000-01-01)
static inline DateADT nanosToPostgresDate(int64_t nanos) {
    constexpr int64_t NANOS_PER_DAY = 86400000000000LL;
    int64_t days = nanos / NANOS_PER_DAY;
    return static_cast<DateADT>(days);
}

static inline int64_t postgresTimestampToNanos(Timestamp ts) {
    constexpr int64_t POSTGRES_EPOCH_OFFSET_USEC = INT64CONST(946684800000000);
    int64_t micros = ts + POSTGRES_EPOCH_OFFSET_USEC;
    return micros * 1000;
}

int64_t runtime::DateRuntime::subtractMonths(int64_t date, int64_t months) {
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("date interval month arithmetic not yet implemented"),
             errhint("Use day-based intervals instead")));
    return 0;
}

int64_t runtime::DateRuntime::addMonths(int64_t date, int64_t months) {
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("date interval month arithmetic not yet implemented"),
             errhint("Use day-based intervals instead")));
    return 0;
}

int64_t runtime::DateRuntime::extractYear(int64_t nanos) {
    PGX_LOG(RUNTIME, DEBUG, "extractYear called with nanos=%ld", nanos);

    DateADT pg_date = nanosToPostgresDate(nanos);
    PGX_LOG(RUNTIME, DEBUG, "converted to pg_date=%d", pg_date);

    int jd = pg_date + POSTGRES_EPOCH_JDATE;
    PGX_LOG(RUNTIME, DEBUG, "julian date=%d", jd);

    int year, month, day;
    j2date(jd, &year, &month, &day);

    PGX_LOG(RUNTIME, DEBUG, "extracted year=%d", year);

    return year;
}

int64_t runtime::DateRuntime::extractMonth(int64_t nanos) {
    DateADT pg_date = nanosToPostgresDate(nanos);

    int jd = pg_date + POSTGRES_EPOCH_JDATE;
    int year, month, day;
    j2date(jd, &year, &month, &day);

    return month;
}

int64_t runtime::DateRuntime::extractDay(int64_t nanos) {
    DateADT pg_date = nanosToPostgresDate(nanos);

    int jd = pg_date + POSTGRES_EPOCH_JDATE;
    int year, month, day;
    j2date(jd, &year, &month, &day);

    return day;
}

int64_t runtime::DateRuntime::ExtractFromDate(VarLen32 field, int64_t date) {
    uint32_t len = field.getLen();
    char* data = field.data();

    PGX_LOG(RUNTIME, DEBUG, "ExtractFromDate called with field='%.*s', date=%ld",
            static_cast<int>(len), data, date);

    if (len == 4 && strncmp(data, "year", 4) == 0) {
        return extractYear(date);
    } else if (len == 5 && strncmp(data, "month", 5) == 0) {
        return extractMonth(date);
    } else if (len == 3 && strncmp(data, "day", 3) == 0) {
        return extractDay(date);
    } else {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("unsupported EXTRACT field: %.*s", static_cast<int>(len), data),
                 errhint("Supported fields are: year, month, day")));
        return 0;
    }
}
