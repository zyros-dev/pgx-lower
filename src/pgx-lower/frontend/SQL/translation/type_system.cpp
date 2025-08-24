// PostgreSQL headers
extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
}

#include "pgx-lower/frontend/SQL/translation/type_system.h"
#include "pgx-lower/utility/logging.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"

namespace pgx_lower::frontend::sql {

PostgreSQLTypeMapper::PostgreSQLTypeMapper(::mlir::MLIRContext& context)
    : context_(context) {}

int32_t PostgreSQLTypeMapper::extractCharLength(int32_t typmod) {
    return typmod >= 0 ? typmod - 4 : 255; // PostgreSQL typmod encoding
}

std::pair<int32_t, int32_t> PostgreSQLTypeMapper::extractNumericInfo(int32_t typmod) {
    if (typmod < 0) {
        // PostgreSQL default for unconstrained NUMERIC
        return {-1, -1};
    }

    // Remove VARHDRSZ offset
    int32_t tmp = typmod - 4;

    // Extract precision and scale
    int32_t precision = (tmp >> 16) & 0xFFFF;
    int32_t scale = tmp & 0xFFFF;

    if (precision < 1 || precision > 1000) {
        PGX_WARNING("Invalid NUMERIC precision: " + std::to_string(precision) + " from typmod "
                    + std::to_string(typmod));
        return {38, 0}; // Safe default
    }

    if (scale < 0 || scale > precision) {
        PGX_WARNING("Invalid NUMERIC scale: " + std::to_string(scale) + " for precision " + std::to_string(precision));
        return {precision, 0}; // Use precision, zero scale
    }

    return {precision, scale};
}

mlir::db::TimeUnitAttr PostgreSQLTypeMapper::extractTimestampPrecision(int32_t typmod) {
    if (typmod < 0) {
        return mlir::db::TimeUnitAttr::microsecond;
    }

    switch (typmod) {
    case 0: return mlir::db::TimeUnitAttr::second;
    case 1:
    case 2:
    case 3: return mlir::db::TimeUnitAttr::millisecond;
    case 4:
    case 5:
    case 6: return mlir::db::TimeUnitAttr::microsecond;
    case 7:
    case 8:
    case 9: return mlir::db::TimeUnitAttr::nanosecond;
    default:
        PGX_WARNING("Invalid TIMESTAMP precision: " + std::to_string(typmod) + ", defaulting to microsecond");
        return mlir::db::TimeUnitAttr::microsecond;
    }
}

::mlir::Type PostgreSQLTypeMapper::mapPostgreSQLType(Oid typeOid, int32_t typmod) {
    switch (typeOid) {
    case INT4OID: return mlir::IntegerType::get(&context_, 32);
    case INT8OID: return mlir::IntegerType::get(&context_, 64);
    case INT2OID: return mlir::IntegerType::get(&context_, 16);
    case FLOAT4OID: return mlir::Float32Type::get(&context_);
    case FLOAT8OID: return mlir::Float64Type::get(&context_);
    case BOOLOID: return mlir::IntegerType::get(&context_, 1);
    case TEXTOID:
    case VARCHAROID: return mlir::db::StringType::get(&context_);
    case BPCHAROID: {
        int32_t maxlen = extractCharLength(typmod);
        return mlir::db::CharType::get(&context_, maxlen);
    }
    case NUMERICOID: {
        auto [precision, scale] = extractNumericInfo(typmod);
        return mlir::db::DecimalType::get(&context_, precision, scale);
    }
    case DATEOID: return mlir::db::DateType::get(&context_, mlir::db::DateUnitAttr::day);
    case TIMESTAMPOID: {
        mlir::db::TimeUnitAttr timeUnit = extractTimestampPrecision(typmod);
        return mlir::db::TimestampType::get(&context_, timeUnit);
    }

    default:
        PGX_WARNING("Unknown PostgreSQL type OID: " + std::to_string(typeOid) + ", defaulting to i32");
        return mlir::IntegerType::get(&context_, 32);
    }
}

} // namespace pgx_lower::frontend::sql