#include <cstring>
#include <array>

#include "lingodb/runtime/RuntimeSpecifications.h"
#include "lingodb/runtime/helpers.h"
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "catalog/pg_type_d.h"
}

namespace runtime {

size_t get_physical_size(uint32_t type_oid) {
    if (type_oid == DATEOID || type_oid == TIMESTAMPOID || type_oid == INTERVALOID) {
        type_oid = INT8OID;
    }

    switch (type_oid) {
    case BOOLOID: return 1;
    case INT2OID: return 2;
    case INT4OID: return 4;
    case INT8OID: return 8;
    case FLOAT4OID: return 4;
    case FLOAT8OID: return 8;
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID:
    case BYTEAOID: return 16;
    case NUMERICOID: return sizeof(NumericDatumCarrier);
    default:
        PGX_ERROR("get_physical_size: Unsupported PostgreSQL type OID: %u", type_oid);
        throw std::runtime_error("Unsupported PostgreSQL type OID");
    }
}

PhysicalType get_physical_type(uint32_t type_oid) {
    if (type_oid == DATEOID || type_oid == TIMESTAMPOID || type_oid == INTERVALOID) {
        type_oid = INT8OID;
    }

    switch (type_oid) {
    case BOOLOID: return PhysicalType::BOOL;
    case INT2OID: return PhysicalType::INT16;
    case INT4OID: return PhysicalType::INT32;
    case INT8OID: return PhysicalType::INT64;
    case FLOAT4OID: return PhysicalType::FLOAT32;
    case FLOAT8OID: return PhysicalType::FLOAT64;
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID:
    case BYTEAOID: return PhysicalType::VARLEN32;
    case NUMERICOID: return PhysicalType::NUMERIC_DATUM;
    default:
        PGX_ERROR("get_physical_size: Unsupported PostgreSQL type OID: %u", type_oid);
        throw std::runtime_error("Unsupported PostgreSQL type OID");
    }
}

NumericDatumCarrier numeric_datum_to_carrier(Datum datum) {
    return static_cast<NumericDatumCarrier>(datum);
}

Datum numeric_datum_from_carrier(NumericDatumCarrier carrier) {
    return static_cast<Datum>(carrier);
}

void store_numeric_datum_carrier(uint8_t* dest, Datum datum) {
    const auto carrier = numeric_datum_to_carrier(datum);
    memcpy(dest, &carrier, sizeof(carrier));
}

NumericDatumCarrier load_numeric_datum_carrier(const uint8_t* src) {
    NumericDatumCarrier carrier = 0;
    memcpy(&carrier, src, sizeof(carrier));
    return carrier;
}

size_t extract_varlen32_string(const uint8_t* varlen32_data, char* dest, size_t max_len) {
    uint8_t dummy = 0;
    VarLen32 decoded(&dummy, 0);
    static_assert(sizeof(decoded) == 16, "VarLen32 layout changed");
    memcpy(&decoded, varlen32_data, sizeof(decoded));

    const size_t len = decoded.getLen();
    const size_t copy_len = (len > max_len) ? max_len : len;

    if (copy_len > 0) {
        memcpy(dest, decoded.getPtr(), copy_len);
    }
    dest[copy_len] = '\0';
    return len;
}

} // namespace runtime
