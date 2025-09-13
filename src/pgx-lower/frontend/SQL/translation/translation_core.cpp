#include "translator_internals.h"

namespace postgresql_ast {

// Translation core implementation - type system and constant translation functionality

#ifdef POSTGRESQL_EXTENSION
// PostgreSQL headers for text handling
extern "C" {
#include "postgres.h"
#include "utils/builtins.h"
#include "fmgr.h"
}
#endif

using namespace pgx_lower::frontend::sql::constants;

std::pair<int32_t, int32_t> PostgreSQLTypeMapper::extractNumericInfo(int32_t typmod) {
    if (typmod < 0) {
        // PostgreSQL default for unconstrained NUMERIC
        return {-1, -1};
    }

    // Remove VARHDRSZ offset
    int32_t tmp = typmod - POSTGRESQL_VARHDRSZ;

    // Extract precision and scale
    int32_t precision = (tmp >> NUMERIC_PRECISION_SHIFT) & NUMERIC_PRECISION_MASK;
    int32_t scale = tmp & NUMERIC_SCALE_MASK;

    if (precision < MIN_NUMERIC_PRECISION || precision > MAX_NUMERIC_PRECISION) {
        PGX_WARNING("Invalid NUMERIC precision: %d from typmod %d", precision, typmod);
        return {MAX_NUMERIC_PRECISION, DEFAULT_NUMERIC_SCALE}; // Safe default
    }

    if (scale < 0 || scale > precision) {
        PGX_WARNING("Invalid NUMERIC scale: %d for precision %d", scale, precision);
        return {precision, DEFAULT_NUMERIC_SCALE}; // Use precision, zero scale
    }

    return {precision, scale};
    }

int32_t PostgreSQLTypeMapper::extractVarcharLength(int32_t typmod) {
    if (typmod < 0) {
        return -1; // No length constraint
    }
    // PostgreSQL stores varchar length as (typmod - 4)
    return typmod - 4;
}

mlir::db::TimeUnitAttr PostgreSQLTypeMapper::extractTimestampPrecision(int32_t typmod) {
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

::mlir::Type PostgreSQLTypeMapper::mapPostgreSQLType(Oid typeOid, int32_t typmod, bool nullable) {
    ::mlir::Type baseType;
    
    switch (typeOid) {
    case INT4OID: baseType = mlir::IntegerType::get(&context_, INT4_BIT_WIDTH); break;
    case INT8OID: baseType = mlir::IntegerType::get(&context_, INT8_BIT_WIDTH); break;
    case INT2OID: baseType = mlir::IntegerType::get(&context_, INT2_BIT_WIDTH); break;
    case FLOAT4OID: baseType = mlir::Float32Type::get(&context_); break;
    case FLOAT8OID: baseType = mlir::Float64Type::get(&context_); break;
    case BOOLOID: baseType = mlir::IntegerType::get(&context_, BOOL_BIT_WIDTH); break;
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: baseType = mlir::db::StringType::get(&context_); break;
    case NUMERICOID: {
        auto [precision, scale] = extractNumericInfo(typmod);
        baseType = mlir::db::DecimalType::get(&context_, precision, scale);
        break;
    }
    case DATEOID: baseType = mlir::db::DateType::get(&context_, mlir::db::DateUnitAttr::day); break;
    case TIMESTAMPOID: {
        mlir::db::TimeUnitAttr timeUnit = extractTimestampPrecision(typmod);
        baseType = mlir::db::TimestampType::get(&context_, timeUnit);
        break;
    }

    default:
        PGX_WARNING(("Unknown PostgreSQL type OID: " + std::to_string(typeOid) + ", defaulting to i32").c_str());
        baseType = mlir::IntegerType::get(&context_, INT4_BIT_WIDTH);
        break;
    }
    
    // Wrap in nullable type if column is nullable
    if (nullable) {
        return mlir::db::NullableType::get(&context_, baseType);
    }
    return baseType;
}

auto translateConst(Const* constNode, ::mlir::OpBuilder& builder, ::mlir::MLIRContext& context) -> ::mlir::Value {
    if (!constNode) {
        PGX_ERROR("Invalid Const parameters");
        return nullptr;
    }

    if (constNode->constisnull) {
        auto nullType = mlir::db::NullableType::get(&context, mlir::IntegerType::get(&context, INT4_BIT_WIDTH));
        return builder.create<mlir::db::NullOp>(builder.getUnknownLoc(), nullType);
    }

    // Map PostgreSQL type to MLIR type
    PostgreSQLTypeMapper typeMapper(context);
    auto mlirType = typeMapper.mapPostgreSQLType(constNode->consttype, constNode->consttypmod);

    switch (constNode->consttype) {
    case INT4OID: {
        int32_t val = static_cast<int32_t>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val, mlirType);
    }
    case INT8OID: {
        int64_t val = static_cast<int64_t>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val, mlirType);
    }
    case INT2OID: {
        int16_t val = static_cast<int16_t>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val, mlirType);
    }
    case FLOAT4OID: {
        float val = *reinterpret_cast<float*>(&constNode->constvalue);
        return builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(),
                                                            llvm::APFloat(val),
                                                            mlirType.cast<mlir::FloatType>());
    }
    case FLOAT8OID: {
        double val = *reinterpret_cast<double*>(&constNode->constvalue);
        return builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(),
                                                            llvm::APFloat(val),
                                                            mlirType.cast<mlir::FloatType>());
    }
    case BOOLOID: {
        bool val = static_cast<bool>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val ? BOOL_TRUE_VALUE : BOOL_FALSE_VALUE, mlirType);
    }
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: {
        // For string constants, constvalue is a pointer to the text data
        // In psql, text values are stored as varlena structures
        #ifdef POSTGRESQL_EXTENSION
        if (constNode->constvalue) {
            text* textval = DatumGetTextP(constNode->constvalue);
            char* str = VARDATA(textval);
            int len = VARSIZE(textval) - VARHDRSZ;
            std::string stringValue(str, len);
            
            return builder.create<mlir::db::ConstantOp>(
                builder.getUnknownLoc(),
                mlirType,
                builder.getStringAttr(stringValue));
        } else {
            return builder.create<mlir::db::ConstantOp>(
                builder.getUnknownLoc(),
                mlirType,
                builder.getStringAttr(""));
        }
        #else
        const char* str = reinterpret_cast<const char*>(constNode->constvalue);
        if (str) {
            return builder.create<mlir::db::ConstantOp>(
                builder.getUnknownLoc(),
                mlirType,
                builder.getStringAttr(str));
        } else {
            return builder.create<mlir::db::ConstantOp>(
                builder.getUnknownLoc(),
                mlirType,
                builder.getStringAttr(""));
        }
        #endif
    }
    default:
        PGX_WARNING("Unsupported constant type: %d", constNode->consttype);
        // Default to i32 zero
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), DEFAULT_FALLBACK_INT_VALUE, builder.getI32Type());
    }
}

} // namespace postgresql_ast