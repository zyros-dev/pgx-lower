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
extern "C" {
#include "utils/numeric.h"
#include "utils/date.h"
#include "datatype/timestamp.h"
#include "utils/timestamp.h"
#include "utils/builtins.h"
}
#endif

using namespace pgx_lower::frontend::sql::constants;

std::pair<int32_t, int32_t> PostgreSQLTypeMapper::extract_numeric_info(const int32_t typmod) {
    PGX_IO(AST_TRANSLATE);
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

int32_t PostgreSQLTypeMapper::extract_varchar_length(int32_t typmod) {
    PGX_IO(AST_TRANSLATE);
    if (typmod < 0) {
        return -1; // No length constraint
    }
    // PostgreSQL stores varchar length as (typmod - 4)
    return typmod - 4;
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

auto PostgreSQLTypeMapper::map_postgre_sqltype(const Oid type_oid, const int32_t typmod, const bool nullable) const
    -> mlir::Type {
    PGX_IO(AST_TRANSLATE);

    auto wrap_nullable = [&](auto val) -> mlir::Type {
        if (nullable)
            return mlir::db::NullableType::get(&context_, val);
        return val;
    };

    switch (type_oid) {
    case INT4OID: return wrap_nullable(mlir::IntegerType::get(&context_, INT4_BIT_WIDTH));
    case INT8OID: return wrap_nullable(mlir::IntegerType::get(&context_, INT8_BIT_WIDTH));
    case INT2OID: return wrap_nullable(mlir::IntegerType::get(&context_, INT2_BIT_WIDTH));
    case FLOAT4OID: return wrap_nullable(mlir::Float32Type::get(&context_));
    case FLOAT8OID: return wrap_nullable(mlir::Float64Type::get(&context_));
    case BOOLOID: return wrap_nullable(mlir::IntegerType::get(&context_, BOOL_BIT_WIDTH));
    // TODO: NV BPCHAROID is supposed to map to a !db.char<X>, but its kind of high effort to add. so I'm just
    //       mapping both of them to strings for the time being.
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: return wrap_nullable(mlir::db::StringType::get(&context_));
    case NUMERICOID: {
        auto [precision, scale] = extract_numeric_info(typmod);
        return wrap_nullable(mlir::db::DecimalType::get(&context_, precision, scale));
    }
    case DATEOID: return wrap_nullable(mlir::db::DateType::get(&context_, mlir::db::DateUnitAttr::day));
    case TIMESTAMPOID: {
        const auto timeUnit = extract_timestamp_precision(typmod);
        return wrap_nullable(mlir::db::TimestampType::get(&context_, timeUnit));
    }
    case INTERVALOID: {
        return wrap_nullable(mlir::db::IntervalType::get(&context_, mlir::db::IntervalUnitAttr::daytime));
    }
    default: {
        PGX_ERROR("Unknown PostgreSQL type OID: %d", type_oid);
        throw std::runtime_error("Unknown PostgreSQL type OID");
    }
    }
}

auto translate_const(Const* constNode, mlir::OpBuilder& builder, mlir::MLIRContext& context) -> mlir::Value {
    PGX_IO(AST_TRANSLATE);
    if (!constNode) {
        PGX_ERROR("Invalid Const parameters");
        throw std::runtime_error("Invalid const parameters");
    }

    if (constNode->constisnull) {
        auto nullType = mlir::db::NullableType::get(&context, mlir::IntegerType::get(&context, INT4_BIT_WIDTH));
        return builder.create<mlir::db::NullOp>(builder.getUnknownLoc(), nullType);
    }

    const auto type_mapper = PostgreSQLTypeMapper(context);
    // Only pass the value for non-null constants
    const auto mlirType = type_mapper.map_postgre_sqltype(constNode->consttype, constNode->consttypmod, false);

    switch (constNode->consttype) {
    case BOOLOID: {
        const bool val = static_cast<bool>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(),
                                                          val ? BOOL_TRUE_VALUE : BOOL_FALSE_VALUE, mlirType);
    }
    case INT2OID: {
        const int16_t val = static_cast<int16_t>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val, mlirType);
    }
    case INT4OID: {
        const int32_t val = static_cast<int32_t>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val, mlirType);
    }
    case INT8OID: {
        const int64_t val = static_cast<int64_t>(constNode->constvalue);
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), val, mlirType);
    }
    case FLOAT4OID: {
        // goofy, C++ doesn't support float32_t and float64_t until C++23... we're on 20. unsure of how to handle this
        // tbh
        const float val = *reinterpret_cast<float*>(&constNode->constvalue);
        return builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(val),
                                                            mlir::cast<mlir::FloatType>(mlirType));
    }
    case FLOAT8OID: {
        const double val = *reinterpret_cast<double*>(&constNode->constvalue);
        return builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(val),
                                                            mlir::cast<mlir::FloatType>(mlirType));
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
        const int64_t days = static_cast<int64_t>(static_cast<int32_t>(constNode->constvalue));
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getI64IntegerAttr(days));
    }
    case TIMESTAMPOID: {
#ifdef POSTGRESQL_EXTENSION
        // Postgres hands us the time as a int64_t, but lingodb stores it as a string. We have two options here...
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
        // TODO Don't, thanks.
        // Convert all intervals to daytime representation for column homogeneity
        const auto* interval = DatumGetIntervalP(constNode->constvalue);

        int64_t totalMicroseconds = interval->time; // Start with time component
        totalMicroseconds += static_cast<int64_t>(interval->day) * USECS_PER_DAY;

        // Convert months to microseconds using the standard approximation
        if (interval->month != 0) {
            int64_t monthMicroseconds = static_cast<int64_t>(interval->month * AVERAGE_DAYS_PER_MONTH * USECS_PER_DAY);
            totalMicroseconds += monthMicroseconds;
        }

        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType,
                                                    builder.getI64IntegerAttr(totalMicroseconds));
#else
        // For unit tests, assume the value is already in microseconds
        int64_t microseconds = static_cast<int64_t>(constNode->constvalue);
        return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType,
                                                    builder.getI64IntegerAttr(microseconds));
#endif
    }
    case TEXTOID:
    case VARCHAROID:
    case BPCHAROID: {
        // For string constants, constvalue is a pointer to the text data
        // In psql, text values are stored as varlena structures
#ifdef POSTGRESQL_EXTENSION
        if (constNode->constvalue) {
            auto* textval = DatumGetTextP(constNode->constvalue);
            char* str = VARDATA(textval);
            int len = VARSIZE(textval) - VARHDRSZ;
            std::string string_value(str, len);

            return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType,
                                                        builder.getStringAttr(string_value));
        } else {
            return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), mlirType, builder.getStringAttr(""));
        }
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