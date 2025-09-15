#pragma once

#include <cstdint>
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "utils/numeric.h"
#include "utils/date.h"
#include "utils/timestamp.h"
}
#endif

namespace runtime {
struct TableBuilder; // fwd decl
}

namespace pgx_lower::runtime {

template<typename T>
inline Datum toDatum(T value) = delete;

template<>
inline Datum toDatum<int8_t>(const int8_t v) {
    return Int16GetDatum(static_cast<int16_t>(v));
}
template<>
inline Datum toDatum<int16_t>(const int16_t v) {
    return Int16GetDatum(v);
}
template<>
inline Datum toDatum<int32_t>(const int32_t v) {
    return Int32GetDatum(v);
}
template<>
inline Datum toDatum<int64_t>(const int64_t v) {
    return Int64GetDatum(v);
} // Also handles Timestamp

template<>
inline Datum toDatum<bool>(const bool v) {
    return BoolGetDatum(v);
}

template<>
inline Datum toDatum<float>(const float v) {
    return Float4GetDatum(v);
}
template<>
inline Datum toDatum<double>(const double v) {
    return Float8GetDatum(v);
}

template<>
inline Datum toDatum<::runtime::VarLen32>(::runtime::VarLen32 v) {
    return PointerGetDatum(v.data());
}

template<>
inline Datum toDatum<Numeric>(Numeric v) {
    return NumericGetDatum(v);
}
template<>
inline Datum toDatum<Interval*>(Interval* v) {
    return IntervalPGetDatum(v);
}


template<typename T>
constexpr Oid getTypeOid() = delete;

template<>
constexpr Oid getTypeOid<int8_t>() {
    return INT2OID;
}
template<>
constexpr Oid getTypeOid<int16_t>() {
    return INT2OID;
}
template<>
constexpr Oid getTypeOid<int32_t>() {
    return INT4OID;
}
template<>
constexpr Oid getTypeOid<int64_t>() {
    return INT8OID;
}

template<>
constexpr Oid getTypeOid<bool>() {
    return BOOLOID;
}

template<>
constexpr Oid getTypeOid<float>() {
    return FLOAT4OID;
}
template<>
constexpr Oid getTypeOid<double>() {
    return FLOAT8OID;
}

template<>
constexpr Oid getTypeOid<::runtime::VarLen32>() {
    return TEXTOID;
}

template<>
constexpr Oid getTypeOid<Numeric>() {
    return NUMERICOID;
}
template<>
constexpr Oid getTypeOid<Interval*>() {
    return INTERVALOID;
}

// Datum-to-Type converters (for field extraction)
template<typename T>
inline T fromDatum(const Datum value, const Oid typeOid) = delete;

// Integer conversions - only allow exact matches and safe widening
template<>
inline int16_t fromDatum<int16_t>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case INT2OID: return DatumGetInt16(value);
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to int16");
    }
}

template<>
inline int32_t fromDatum<int32_t>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case INT4OID: return DatumGetInt32(value);
    case DATEOID: return DatumGetDateADT(value); // DateADT is int32
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to int32");
    }
}

template<>
inline int64_t fromDatum<int64_t>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case INT8OID: return DatumGetInt64(value);
    case TIMESTAMPOID: return DatumGetTimestamp(value);
    case INTERVALOID: {
        const auto* interval = DatumGetIntervalP(value);
        int64_t totalMicroseconds = interval->time + (static_cast<int64_t>(interval->day) * USECS_PER_DAY);

        // TODO: NV This is obviously not good
        // Convert months to microseconds
        if (interval->month != 0) {
            int64_t monthMicroseconds = static_cast<int64_t>(
                interval->month * frontend::sql::constants::AVERAGE_DAYS_PER_MONTH * USECS_PER_DAY);
            totalMicroseconds += monthMicroseconds;
        }
        return totalMicroseconds;
    }
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to int64");
    }
}

template<>
inline bool fromDatum<bool>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case BOOLOID: return DatumGetBool(value);
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to bool");
    }
}

template<>
inline float fromDatum<float>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case FLOAT4OID: return DatumGetFloat4(value);
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to float");
    }
}

template<>
inline double fromDatum<double>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case FLOAT8OID: return DatumGetFloat8(value);
    default: {
        PGX_ERROR("Cannot convert type OID %d to double", typeOid);
        throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to double");
    }
    }
}

template<>
inline Numeric fromDatum<Numeric>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case NUMERICOID: return DatumGetNumeric(value);
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to Numeric");
    }
}

template<>
inline Interval* fromDatum<Interval*>(const Datum value, const Oid typeOid) {
    switch (typeOid) {
    case INTERVALOID: return DatumGetIntervalP(value);
    default: throw std::runtime_error("Cannot convert type OID " + std::to_string(typeOid) + " to Interval");
    }
}

template<typename T>
void table_builder_add(void* builder, bool is_valid, T value);

template<typename T>
T extract_field(int32_t field_index, bool* is_null);

} // namespace pgx_lower::runtime