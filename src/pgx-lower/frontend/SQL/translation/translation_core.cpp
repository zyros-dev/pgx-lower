// Translation core implementation - included directly into postgresql_ast_translator.cpp
// Contains type system and constant translation functionality

using namespace pgx_lower::frontend::sql::constants;

class PostgreSQLTypeMapper {
public:
    explicit PostgreSQLTypeMapper(::mlir::MLIRContext& context)
        : context_(context) {}

    int32_t extractCharLength(int32_t typmod) {
    return typmod >= 0 ? typmod - POSTGRESQL_VARHDRSZ : DEFAULT_VARCHAR_LENGTH; // PostgreSQL typmod encoding
    }

    std::pair<int32_t, int32_t> extractNumericInfo(int32_t typmod) {
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
        PGX_WARNING("Invalid NUMERIC precision: " + std::to_string(precision) + " from typmod "
                    + std::to_string(typmod));
        return {MAX_NUMERIC_PRECISION, DEFAULT_NUMERIC_SCALE}; // Safe default
    }

    if (scale < 0 || scale > precision) {
        PGX_WARNING("Invalid NUMERIC scale: " + std::to_string(scale) + " for precision " + std::to_string(precision));
        return {precision, DEFAULT_NUMERIC_SCALE}; // Use precision, zero scale
    }

    return {precision, scale};
    }

    mlir::db::TimeUnitAttr extractTimestampPrecision(int32_t typmod) {
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
        PGX_WARNING("Invalid TIMESTAMP precision: " + std::to_string(typmod) + ", defaulting to microsecond");
        return mlir::db::TimeUnitAttr::microsecond;
    }
    }

    ::mlir::Type mapPostgreSQLType(unsigned int typeOid, int32_t typmod) {
    switch (typeOid) {
    case INT4OID: return mlir::IntegerType::get(&context_, INT4_BIT_WIDTH);
    case INT8OID: return mlir::IntegerType::get(&context_, INT8_BIT_WIDTH);
    case INT2OID: return mlir::IntegerType::get(&context_, INT2_BIT_WIDTH);
    case FLOAT4OID: return mlir::Float32Type::get(&context_);
    case FLOAT8OID: return mlir::Float64Type::get(&context_);
    case BOOLOID: return mlir::IntegerType::get(&context_, BOOL_BIT_WIDTH);
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
        return mlir::IntegerType::get(&context_, INT4_BIT_WIDTH);
    }
    }

private:
    ::mlir::MLIRContext& context_;
};

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
    default:
        PGX_WARNING("Unsupported constant type: " + std::to_string(constNode->consttype));
        // Default to i32 zero
        return builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), DEFAULT_FALLBACK_INT_VALUE, builder.getI32Type());
    }
}