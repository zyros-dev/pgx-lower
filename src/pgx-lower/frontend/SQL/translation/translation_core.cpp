#include "translator_internals.h"

#include <utils/timestamp.h>
extern "C" {
#include "nodes/primnodes.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"

#include <string>

namespace postgresql_ast {

#ifdef POSTGRESQL_EXTENSION
extern "C" {}
#endif

using namespace pgx_lower::frontend::sql::constants;

auto PostgreSQLTypeMapper::map_postgre_sqltype(const Oid type_oid, const int32_t typmod, const Oid collation,
                                               const bool nullable) const -> mlir::Type {
    PGX_IO(AST_TRANSLATE);
    const auto nullability = nullable ? mlir::db::PgNullability::Maybe : mlir::db::PgNullability::Never;
    const auto pgCollation = static_cast<mlir::db::PgOid>(collation);

    switch (type_oid) {
    case BOOLOID: return mlir::db::PgBoolType::get(&context_, nullability);
    case INT2OID: return mlir::db::PgInt2Type::get(&context_, nullability);
    case INT4OID: return mlir::db::PgInt4Type::get(&context_, nullability);
    case INT8OID: return mlir::db::PgInt8Type::get(&context_, nullability);
    case FLOAT4OID: return mlir::db::PgFloat4Type::get(&context_, nullability);
    case FLOAT8OID: return mlir::db::PgFloat8Type::get(&context_, nullability);
    case TEXTOID: return mlir::db::PgTextType::get(&context_, pgCollation, nullability);
    case VARCHAROID: return mlir::db::PgVarcharType::get(&context_, typmod, pgCollation, nullability);
    case BPCHAROID: return mlir::db::PgBpcharType::get(&context_, typmod, pgCollation, nullability);
    case BYTEAOID: {
        PGX_ERROR("BYTEA is not supported as a PostgreSQL value type in the primitive mapper");
        throw std::runtime_error("Unsupported PostgreSQL BYTEA value type");
    }
    case NUMERICOID: return mlir::db::PgNumericType::get(&context_, typmod, nullability);
    case DATEOID: return mlir::db::PgDateType::get(&context_, nullability);
    case TIMESTAMPOID: return mlir::db::PgTimestampType::get(&context_, typmod, nullability);
    case INTERVALOID: return mlir::db::PgIntervalType::get(&context_, typmod, nullability);
    default: {
        PGX_ERROR("Unknown PostgreSQL type OID: %d", type_oid);
        throw std::runtime_error("Unknown PostgreSQL type OID");
    }
    }
}

std::pair<int32_t, int32_t> PostgreSQLTypeMapper::extract_numeric_info(const int32_t typmod) {
    PGX_IO(AST_TRANSLATE);
    if (typmod < 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "No typmod specified - using flexible precision for numeric type");
        return {MAX_NUMERIC_PRECISION, MAX_NUMERIC_UNCONSTRAINED_SCALE};
    }

    const int32_t tmp = typmod - POSTGRESQL_VARHDRSZ;
    int32_t precision = (tmp >> NUMERIC_PRECISION_SHIFT) & NUMERIC_PRECISION_MASK;
    const int32_t scale = tmp & NUMERIC_SCALE_MASK;

    if (precision < MIN_NUMERIC_PRECISION || precision > MAX_NUMERIC_PRECISION) {
        PGX_WARNING("Invalid NUMERIC precision: %d from typmod %d", precision, typmod);
        return {MAX_NUMERIC_PRECISION, MAX_NUMERIC_UNCONSTRAINED_SCALE};
    }

    if (scale < 0 || scale > precision) {
        PGX_WARNING("Invalid NUMERIC scale: %d for precision %d", scale, precision);
        return {precision, MAX_NUMERIC_UNCONSTRAINED_SCALE};
    }

    return {std::min(precision, MAX_NUMERIC_PRECISION), std::min(scale, MAX_NUMERIC_UNCONSTRAINED_SCALE)};
}

Oid PostgreSQLTypeMapper::map_mlir_type_to_oid(mlir::Type mlir_type) {
    if (const auto nullable_type = mlir::dyn_cast<mlir::db::NullableType>(mlir_type)) {
        if (mlir::db::isPgValueType(nullable_type.getType())) {
            PGX_WARNING("Legacy NullableType cannot define PostgreSQL identity for PG semantic payloads");
            return InvalidOid;
        }
    }

    if (mlir::db::isPgValueType(mlir_type)) {
        return mlir::db::getPgTypeOid(mlir_type);
    }

    PGX_WARNING("Unable to map non-PostgreSQL MLIR type to PostgreSQL OID, returning InvalidOid");
    return InvalidOid;
}

mlir::db::TimeUnitAttr PostgreSQLTypeMapper::extract_timestamp_precision(const int32_t typmod) {
    PGX_IO(AST_TRANSLATE);
    if (typmod < 0) {
        return mlir::db::TimeUnitAttr::microsecond;
    }

    switch (typmod) {
    case TIMESTAMP_PRECISION_SECOND: return mlir::db::TimeUnitAttr::second;
    case TIMESTAMP_PRECISION_MILLI_MIN:
    case 2:
    case TIMESTAMP_PRECISION_MILLI_MAX: return mlir::db::TimeUnitAttr::millisecond;
    case TIMESTAMP_PRECISION_MICRO_MIN:
    case 5:
    case TIMESTAMP_PRECISION_MICRO_MAX: return mlir::db::TimeUnitAttr::microsecond;
    case TIMESTAMP_PRECISION_NANO_MIN:
    case 8:
    case TIMESTAMP_PRECISION_NANO_MAX: return mlir::db::TimeUnitAttr::nanosecond;
    default:
        PGX_WARNING(("Invalid TIMESTAMP precision: " + std::to_string(typmod) + ", defaulting to microsecond").c_str());
        return mlir::db::TimeUnitAttr::microsecond;
    }
}

int32_t PostgreSQLTypeMapper::extract_varchar_length(const int32_t typmod) {
    PGX_IO(AST_TRANSLATE);
    if (typmod < 0) {
        return -1; // No length constraint
    }
    // PostgreSQL stores varchar length as (typmod - 4)
    return typmod - 4;
}

auto translate_const(Const* constNode, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!constNode) {
        PGX_ERROR("Invalid Const parameters");
        throw std::runtime_error("Invalid const parameters");
    }

    const auto type_mapper = PostgreSQLTypeMapper(context);

    if (constNode->constisnull) {
        auto nullType = type_mapper.map_postgre_sqltype(constNode->consttype, constNode->consttypmod,
                                                        constNode->constcollid, true);
        return builder.create<mlir::db::NullOp>(builder.getUnknownLoc(), nullType);
    }

    const auto mlirType = type_mapper.map_postgre_sqltype(constNode->consttype, constNode->consttypmod,
                                                          constNode->constcollid, false);

    switch (constNode->consttype) {
    case BOOLOID: {
        const bool val = static_cast<bool>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getBoolAttr(val));
    }
    case INT2OID: {
        const int16_t val = static_cast<int16_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getI16IntegerAttr(val));
    }
    case INT4OID: {
        const int32_t val = static_cast<int32_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getI32IntegerAttr(val));
    }
    case INT8OID: {
        const int64_t val = static_cast<int64_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getI64IntegerAttr(val));
    }
    case FLOAT4OID: {
        // goofy, C++ doesn't support float32_t and float64_t until C++23... we're on 20. unsure of how to handle this
        const float val = *reinterpret_cast<float*>(&constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getF32FloatAttr(val));
    }
    case FLOAT8OID: {
        const double val = *reinterpret_cast<double*>(&constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getF64FloatAttr(val));
    }
    case NUMERICOID: {
#ifdef POSTGRESQL_EXTENSION
        // PostgreSQL stores NUMERIC as a pointer to a variable-length structure
        // LingoDB stores decimals as string attributes for exact precision
        // Use PostgreSQL's numeric_out function to get the exact string representation
        const auto numericDatum = constNode->constvalue;
        char* numericStr = DatumGetCString(DirectFunctionCall1(numeric_out, numericDatum));
        const auto numStr = std::string(numericStr);
        pfree(numericStr);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(numStr));
#else
        int64_t val = static_cast<int64_t>(constNode->constvalue);
        std::string numStr = std::to_string(val);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(numStr));
#endif
    }
    case DATEOID: {
        const int32_t days = static_cast<int32_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getI32IntegerAttr(days));
    }
    case TIMESTAMPOID: {
#ifdef POSTGRESQL_EXTENSION
        // Postgres hands us the time as an int64_t, but lingodb stores it as a string. We have two options here...
        // hand lingodb the string and don't worry, or adjust lingodb to handle int64s... I will rather rely on
        // lingodb's solution.
        const Timestamp timestamp = static_cast<Timestamp>(constNode->constvalue);
        char* timestampStr = DatumGetCString(DirectFunctionCall1(timestamp_out, TimestampGetDatum(timestamp)));
        const auto timeStr = std::string(timestampStr);
        pfree(timestampStr);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(timeStr));
#else
        // For unit tests, just pass the microseconds as before
        const int64_t microseconds = static_cast<int64_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType,
                                                    builder.getI64IntegerAttr(microseconds));
#endif
    }
    case INTERVALOID: {
#ifdef POSTGRESQL_EXTENSION
        const auto* interval = DatumGetIntervalP(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(
            builder.getUnknownLoc(), mlirType,
            builder.getArrayAttr({builder.getI64IntegerAttr(interval->time), builder.getI32IntegerAttr(interval->day),
                                  builder.getI32IntegerAttr(interval->month)}));
#else
        int64_t microseconds = static_cast<int64_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType,
                                                    builder.getI64IntegerAttr(microseconds));
#endif
    }
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: {
#ifdef POSTGRESQL_EXTENSION
        if (constNode->constvalue != 0u) {
            const auto* packedText = DatumGetTextPP(constNode->constvalue);
            const auto payloadLength = VARSIZE_ANY_EXHDR(packedText);
            const char* payloadBytes = VARDATA_ANY(packedText);
            const std::string stringValue(payloadBytes, payloadLength);

            PGX_LOG(AST_TRANSLATE, DEBUG, "String constant length=%d, type_oid=%d, typmod=%d", payloadLength,
                    constNode->consttype, constNode->consttypmod);

            return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType,
                                                        builder.getStringAttr(stringValue));
        }
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(""));

#else
        const char* str = reinterpret_cast<const char*>(constNode->constvalue);
        if (str) {
            return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(str));
        } else {
            return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(""));
        }
#endif
    }
    default:
        PGX_ERROR("Unsupported constant type: %d", constNode->consttype);
        throw std::runtime_error("Unsupported constant type");
    }
}

} // namespace postgresql_ast
