#include "lingodb/runtime/RuntimeSpecifications.h"
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
    case NUMERICOID: return 16;
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
    case NUMERICOID: return PhysicalType::DECIMAL128;
    default:
        PGX_ERROR("get_physical_size: Unsupported PostgreSQL type OID: %u", type_oid);
        throw std::runtime_error("Unsupported PostgreSQL type OID");
    }
}

size_t extract_varlen32_string(const uint8_t* i128_data, char* dest, size_t max_len) {
    // VarLen32 i128 layout - TWO cases:
    // Case 1 (lazy flag SET): Runtime pointer-based string from table scan
    //   bytes[0-3]:   len | 0x80000000
    //   bytes[4-7]:   unused
    //   bytes[8-15]:  valid pointer to string data
    // Case 2 (lazy flag CLEAR): MLIR inlined constant from CASE/literal
    //   bytes[0-3]:   len (no flag)
    //   bytes[4-7]:   first 4 bytes of string
    //   bytes[8-15]:  remaining bytes of string

    const uint32_t len_with_flag = *reinterpret_cast<const uint32_t*>(i128_data);
    const bool is_lazy = (len_with_flag & 0x80000000u) != 0;
    const size_t len = len_with_flag & ~0x80000000u;

    // Safety check
    const size_t copy_len = (len > max_len) ? max_len : len;

    if (is_lazy) {
        const char* str_ptr = *reinterpret_cast<char* const*>(i128_data + 8);
        memcpy(dest, str_ptr, copy_len);
    } else {
        const size_t first = (copy_len < 4) ? copy_len : 4;
        memcpy(dest, i128_data + 4, first);
        if (copy_len > 4) {
            memcpy(dest + 4, i128_data + 8, copy_len - 4);
        }
    }

    dest[copy_len] = '\0';
    return len;
}

} // namespace runtime
