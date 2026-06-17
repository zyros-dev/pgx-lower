#include "lingodb/runtime/DateRuntime.h"
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "utils/date.h"
#include "utils/datetime.h"
}

int64_t runtime::DateRuntime::subtractMonths(int64_t, int64_t) {
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("date interval month arithmetic not yet implemented"),
             errhint("Use day-based intervals instead")));
    return 0;
}

int64_t runtime::DateRuntime::addMonths(int64_t, int64_t) {
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("date interval month arithmetic not yet implemented"),
             errhint("Use day-based intervals instead")));
    return 0;
}

int64_t runtime::DateRuntime::extractYear(int64_t date) {
    PGX_LOG(RUNTIME, DEBUG, "extractYear called with date=%ld", date);

    const auto pg_date = static_cast<DateADT>(date);

    const int jd{pg_date + POSTGRES_EPOCH_JDATE};
    PGX_LOG(RUNTIME, DEBUG, "julian date=%d", jd);

    int year{};
    int month{};
    int day{};
    j2date(jd, &year, &month, &day);

    PGX_LOG(RUNTIME, DEBUG, "extracted year=%d", year);

    return year;
}

int64_t runtime::DateRuntime::extractMonth(int64_t date) {
    const auto pg_date = static_cast<DateADT>(date);

    const int jd{pg_date + POSTGRES_EPOCH_JDATE};
    int year{};
    int month{};
    int day{};
    j2date(jd, &year, &month, &day);

    return month;
}

int64_t runtime::DateRuntime::extractDay(int64_t date) {
    const auto pg_date = static_cast<DateADT>(date);

    const int jd{pg_date + POSTGRES_EPOCH_JDATE};
    int year{};
    int month{};
    int day{};
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
    }
    if (len == 5 && strncmp(data, "month", 5) == 0) {
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
