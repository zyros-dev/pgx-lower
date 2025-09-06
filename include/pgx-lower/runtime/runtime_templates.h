#pragma once

#include <cstdint>
#include "lingodb/runtime/helpers.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/numeric.h"
}
#endif

// Using forward declaration to avoid circular dependencies
namespace runtime {
    struct TableBuilder;
}

namespace pgx_lower::runtime {

// Forward declarations - removed g_computed_results as it requires complete type

// ============================================================================
// Type-to-Datum converters
// ============================================================================

template<typename T>
inline Datum toDatum(T value);

// Integer type specializations
template<> inline Datum toDatum<int8_t>(int8_t v) { 
    return Int16GetDatum(static_cast<int16_t>(v));  // PostgreSQL doesn't have INT1
}
template<> inline Datum toDatum<int16_t>(int16_t v) { return Int16GetDatum(v); }
template<> inline Datum toDatum<int32_t>(int32_t v) { return Int32GetDatum(v); }
template<> inline Datum toDatum<int64_t>(int64_t v) { return Int64GetDatum(v); }

// Boolean specialization  
template<> inline Datum toDatum<bool>(bool v) { return BoolGetDatum(v); }

// Floating point specializations
template<> inline Datum toDatum<float>(float v) { return Float4GetDatum(v); }
template<> inline Datum toDatum<double>(double v) { return Float8GetDatum(v); }

// VarLen32 specialization for binary/string data
template<> inline Datum toDatum<::runtime::VarLen32>(::runtime::VarLen32 v) { 
    // TODO: Properly convert VarLen32 to PostgreSQL text/bytea
    // For now, just store as a pointer
    return PointerGetDatum(v.data());
}

// ============================================================================
// Type-to-OID mapping
// ============================================================================

template<typename T>
constexpr Oid getTypeOid();

// Integer type OID mappings
template<> constexpr Oid getTypeOid<int8_t>() { return INT2OID; }  // Map to INT2
template<> constexpr Oid getTypeOid<int16_t>() { return INT2OID; }
template<> constexpr Oid getTypeOid<int32_t>() { return INT4OID; }
template<> constexpr Oid getTypeOid<int64_t>() { return INT8OID; }

// Boolean OID mapping
template<> constexpr Oid getTypeOid<bool>() { return BOOLOID; }

// Floating point OID mappings
template<> constexpr Oid getTypeOid<float>() { return FLOAT4OID; }
template<> constexpr Oid getTypeOid<double>() { return FLOAT8OID; }

// VarLen32 OID mapping (use TEXTOID for now)
template<> constexpr Oid getTypeOid<::runtime::VarLen32>() { return TEXTOID; }

// Datum-to-Type converters (for field extraction)
template<typename T>
inline T fromDatum(Datum value, Oid typeOid);

// Integer conversions with type coercion
template<> inline int16_t fromDatum<int16_t>(Datum value, Oid typeOid) {
    switch (typeOid) {
        case INT2OID: return DatumGetInt16(value);
        case INT4OID: return static_cast<int16_t>(DatumGetInt32(value));
        case INT8OID: return static_cast<int16_t>(DatumGetInt64(value));
        case BOOLOID: return DatumGetBool(value) ? 1 : 0;
        default: return 0;
    }
}

template<> inline int32_t fromDatum<int32_t>(Datum value, Oid typeOid) {
    switch (typeOid) {
        case INT2OID: return static_cast<int32_t>(DatumGetInt16(value));
        case INT4OID: return DatumGetInt32(value);
        case INT8OID: return static_cast<int32_t>(DatumGetInt64(value));
        case BOOLOID: return DatumGetBool(value) ? 1 : 0;
        default: return 0;
    }
}

template<> inline int64_t fromDatum<int64_t>(Datum value, Oid typeOid) {
    switch (typeOid) {
        case INT2OID: return static_cast<int64_t>(DatumGetInt16(value));
        case INT4OID: return static_cast<int64_t>(DatumGetInt32(value));
        case INT8OID: return DatumGetInt64(value);
        case BOOLOID: return DatumGetBool(value) ? 1 : 0;
        default: return 0;
    }
}

// Boolean conversions
template<> inline bool fromDatum<bool>(Datum value, Oid typeOid) {
    switch (typeOid) {
        case BOOLOID: return DatumGetBool(value);
        case INT2OID: return DatumGetInt16(value) != 0;
        case INT4OID: return DatumGetInt32(value) != 0;
        case INT8OID: return DatumGetInt64(value) != 0;
        default: return false;
    }
}

// Floating point conversions
template<> inline float fromDatum<float>(Datum value, Oid typeOid) {
    switch (typeOid) {
        case FLOAT4OID: return DatumGetFloat4(value);
        case FLOAT8OID: return static_cast<float>(DatumGetFloat8(value));
        case INT2OID: return static_cast<float>(DatumGetInt16(value));
        case INT4OID: return static_cast<float>(DatumGetInt32(value));
        case INT8OID: return static_cast<float>(DatumGetInt64(value));
        default: return 0.0f;
    }
}

template<> inline double fromDatum<double>(Datum value, Oid typeOid) {
    switch (typeOid) {
        case FLOAT4OID: return static_cast<double>(DatumGetFloat4(value));
        case FLOAT8OID: return DatumGetFloat8(value);
        case INT2OID: return static_cast<double>(DatumGetInt16(value));
        case INT4OID: return static_cast<double>(DatumGetInt32(value));
        case INT8OID: return static_cast<double>(DatumGetInt64(value));
        default: return 0.0;
    }
}

// ============================================================================
// Forward declarations for template functions (defined in tuple_access.cpp)
// ============================================================================

template<typename T>
void table_builder_add(void* builder, bool is_valid, T value);

template<typename T>
T extract_field(int32_t field_index, bool* is_null);

} // namespace pgx_lower::runtime