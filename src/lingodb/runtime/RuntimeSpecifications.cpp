#include "lingodb/runtime/RuntimeSpecifications.h"

extern "C" {
#include "postgres.h"
#include "catalog/pg_type_d.h"
}

namespace runtime {

size_t get_physical_size(uint32_t type_oid) {
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
    case NUMERICOID: return 16;
    case DATEOID: return 8;
    case TIMESTAMPOID: return 8;
    case INTERVALOID: return 8;
    default: throw std::runtime_error("Unsupported PostgreSQL type OID");
    }
}

PhysicalType get_physical_type(uint32_t type_oid) {
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
    case NUMERICOID: return PhysicalType::DECIMAL128;
    case DATEOID: return PhysicalType::INT64;
    case TIMESTAMPOID: return PhysicalType::INT64;
    case INTERVALOID: return PhysicalType::INT64;
    default: throw std::runtime_error("Unsupported PostgreSQL type OID");
    }
}

} // namespace runtime
