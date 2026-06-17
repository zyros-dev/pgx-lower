#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "lingodb/mlir/Dialect/util/UtilTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#include <cstdint>
#include <limits>

namespace mlir { namespace db {

namespace {
constexpr int32_t kPgTypmodUnconstrained = -1;

auto parseNonPgNullablePayload(AsmParser& parser) -> Type {
    if (parser.parseLess()) {
        return {};
    }
    auto payload = FieldParser<Type>::parse(parser);
    if (failed(payload)) {
        parser.emitError(parser.getCurrentLocation(), "failed to parse NullableType payload type");
        return {};
    }
    if (mlir::db::isPgValueType(*payload)) {
        parser.emitError(parser.getCurrentLocation(), "legacy nullable cannot wrap PostgreSQL semantic types");
        return {};
    }
    if (parser.parseGreater()) {
        return {};
    }
    return *payload;
}

auto parseNullableKeyword(AsmParser& parser) -> FailureOr<PgNullability> {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
        return failure();
    }
    if (keyword != "nullable") {
        return parser.emitError(parser.getCurrentLocation(), "expected 'nullable'");
    }
    return PgNullability::Maybe;
}

auto parseOptionalNullability(AsmParser& parser) -> FailureOr<PgNullability> {
    if (failed(parser.parseOptionalComma())) {
        return PgNullability::Never;
    }
    return parseNullableKeyword(parser);
}

auto parseTypmodValue(AsmParser& parser) -> FailureOr<int32_t> {
    int64_t parsed{};
    if (parser.parseInteger(parsed)) {
        return failure();
    }
    if (parsed < kPgTypmodUnconstrained || parsed > std::numeric_limits<int32_t>::max()) {
        return parser.emitError(parser.getCurrentLocation(), "expected typmod >= -1");
    }
    return static_cast<int32_t>(parsed);
}

auto parsePgOidValue(AsmParser& parser) -> FailureOr<PgOid> {
    uint64_t parsed{};
    if (parser.parseInteger(parsed)) {
        return failure();
    }
    if (parsed > std::numeric_limits<PgOid>::max()) {
        return parser.emitError(parser.getCurrentLocation(), "expected OID-sized unsigned integer");
    }
    return static_cast<PgOid>(parsed);
}

template<typename TypeT>
auto parseNoMetadataPgType(AsmParser& parser) -> Type {
    PgNullability nullability = PgNullability::Never;
    if (succeeded(parser.parseOptionalLess())) {
        auto parsedNullability = parseNullableKeyword(parser);
        if (failed(parsedNullability) || parser.parseGreater()) {
            return Type();
        }
        nullability = *parsedNullability;
    }
    return TypeT::get(parser.getContext(), nullability);
}

void printPgNullability(AsmPrinter& printer, PgNullability nullability) {
    if (nullability == PgNullability::Maybe) {
        printer << "nullable";
    }
}

template<typename TypeT>
void printNoMetadataPgType(TypeT type, AsmPrinter& printer) {
    if (type.getNullability() == PgNullability::Maybe) {
        printer << "<";
        printPgNullability(printer, type.getNullability());
        printer << ">";
    }
}

template<typename TypeT>
auto parseTypmodPgType(AsmParser& parser) -> Type {
    if (parser.parseLess() || parser.parseKeyword("typmod") || parser.parseEqual()) {
        return Type();
    }
    auto typmod = parseTypmodValue(parser);
    if (failed(typmod)) {
        return Type();
    }
    auto nullability = parseOptionalNullability(parser);
    if (failed(nullability) || parser.parseGreater()) {
        return Type();
    }
    return TypeT::get(parser.getContext(), *typmod, *nullability);
}

template<typename TypeT>
void printTypmodPgType(TypeT type, AsmPrinter& printer) {
    printer << "<typmod = " << type.getTypmod();
    if (type.getNullability() == PgNullability::Maybe) {
        printer << ", ";
        printPgNullability(printer, type.getNullability());
    }
    printer << ">";
}

auto parsePgTextType(AsmParser& parser) -> Type {
    if (parser.parseLess() || parser.parseKeyword("collation") || parser.parseEqual()) {
        return Type();
    }
    auto collation = parsePgOidValue(parser);
    if (failed(collation)) {
        return Type();
    }
    auto nullability = parseOptionalNullability(parser);
    if (failed(nullability) || parser.parseGreater()) {
        return Type();
    }
    return PgTextType::get(parser.getContext(), *collation, *nullability);
}

auto parseStringTypmodPgType(AsmParser& parser, bool bpchar) -> Type {
    if (parser.parseLess() || parser.parseKeyword("typmod") || parser.parseEqual()) {
        return Type();
    }
    auto typmod = parseTypmodValue(parser);
    if (failed(typmod) || parser.parseComma() || parser.parseKeyword("collation") || parser.parseEqual()) {
        return Type();
    }
    auto collation = parsePgOidValue(parser);
    if (failed(collation)) {
        return Type();
    }
    auto nullability = parseOptionalNullability(parser);
    if (failed(nullability) || parser.parseGreater()) {
        return Type();
    }
    if (bpchar) {
        return PgBpcharType::get(parser.getContext(), *typmod, *collation, *nullability);
    }
    return PgVarcharType::get(parser.getContext(), *typmod, *collation, *nullability);
}

template<typename TypeT>
void printStringTypmodPgType(TypeT type, AsmPrinter& printer) {
    printer << "<typmod = " << type.getTypmod() << ", collation = " << type.getCollation();
    if (type.getNullability() == PgNullability::Maybe) {
        printer << ", ";
        printPgNullability(printer, type.getNullability());
    }
    printer << ">";
}

} // namespace

Type NullableType::parse(AsmParser& parser) {
    auto payload = parseNonPgNullablePayload(parser);
    if (!payload) {
        return {};
    }
    return NullableType::get(parser.getContext(), payload);
}

void NullableType::print(AsmPrinter& printer) const {
    printer << "<";
    printer.printStrippedAttrOrType(getType());
    printer << ">";
}

Type PgBoolType::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgBoolType>(parser);
}
void PgBoolType::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}
Type PgInt2Type::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgInt2Type>(parser);
}
void PgInt2Type::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}
Type PgInt4Type::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgInt4Type>(parser);
}
void PgInt4Type::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}
Type PgInt8Type::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgInt8Type>(parser);
}
void PgInt8Type::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}
Type PgFloat4Type::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgFloat4Type>(parser);
}
void PgFloat4Type::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}
Type PgFloat8Type::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgFloat8Type>(parser);
}
void PgFloat8Type::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}
Type PgDateType::parse(AsmParser& parser) {
    return parseNoMetadataPgType<PgDateType>(parser);
}
void PgDateType::print(AsmPrinter& printer) const {
    printNoMetadataPgType(*this, printer);
}

Type PgNumericType::parse(AsmParser& parser) {
    return parseTypmodPgType<PgNumericType>(parser);
}
void PgNumericType::print(AsmPrinter& printer) const {
    printTypmodPgType(*this, printer);
}
Type PgTimestampType::parse(AsmParser& parser) {
    return parseTypmodPgType<PgTimestampType>(parser);
}
void PgTimestampType::print(AsmPrinter& printer) const {
    printTypmodPgType(*this, printer);
}
Type PgIntervalType::parse(AsmParser& parser) {
    return parseTypmodPgType<PgIntervalType>(parser);
}
void PgIntervalType::print(AsmPrinter& printer) const {
    printTypmodPgType(*this, printer);
}

Type PgTextType::parse(AsmParser& parser) {
    return parsePgTextType(parser);
}
void PgTextType::print(AsmPrinter& printer) const {
    printer << "<collation = " << getCollation();
    if (getNullability() == PgNullability::Maybe) {
        printer << ", ";
        printPgNullability(printer, getNullability());
    }
    printer << ">";
}
Type PgVarcharType::parse(AsmParser& parser) {
    return parseStringTypmodPgType(parser, false);
}
void PgVarcharType::print(AsmPrinter& printer) const {
    printStringTypmodPgType(*this, printer);
}
Type PgBpcharType::parse(AsmParser& parser) {
    return parseStringTypmodPgType(parser, true);
}
void PgBpcharType::print(AsmPrinter& printer) const {
    printStringTypmodPgType(*this, printer);
}

}} // namespace mlir::db
#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace mlir::db {

bool isPgValueType(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, bool>(type)
        .Case<PgBoolType, PgInt2Type, PgInt4Type, PgInt8Type, PgFloat4Type, PgFloat8Type, PgNumericType, PgDateType,
              PgTimestampType, PgIntervalType, PgTextType, PgVarcharType, PgBpcharType>([](auto) { return true; })
        .Default([](auto) { return false; });
}

PgOid getPgTypeOid(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, PgOid>(type)
        .Case<PgBoolType>([](auto) { return BOOLOID; })
        .Case<PgInt2Type>([](auto) { return INT2OID; })
        .Case<PgInt4Type>([](auto) { return INT4OID; })
        .Case<PgInt8Type>([](auto) { return INT8OID; })
        .Case<PgFloat4Type>([](auto) { return FLOAT4OID; })
        .Case<PgFloat8Type>([](auto) { return FLOAT8OID; })
        .Case<PgNumericType>([](auto) { return NUMERICOID; })
        .Case<PgDateType>([](auto) { return DATEOID; })
        .Case<PgTimestampType>([](auto) { return TIMESTAMPOID; })
        .Case<PgIntervalType>([](auto) { return INTERVALOID; })
        .Case<PgTextType>([](auto) { return TEXTOID; })
        .Case<PgVarcharType>([](auto) { return VARCHAROID; })
        .Case<PgBpcharType>([](auto) { return BPCHAROID; })
        .Default([](auto) { return InvalidOid; });
}

int32_t getPgTypmod(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, int32_t>(type)
        .Case<PgNumericType, PgTimestampType, PgIntervalType, PgVarcharType, PgBpcharType>(
            [](auto typed) { return typed.getTypmod(); })
        .Default([](auto) { return kPgTypmodUnconstrained; });
}

PgOid getPgCollation(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, PgOid>(type)
        .Case<PgTextType, PgVarcharType, PgBpcharType>([](auto typed) { return typed.getCollation(); })
        .Default([](auto) { return InvalidOid; });
}

PgNullability getPgNullability(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, PgNullability>(type)
        .Case<PgBoolType, PgInt2Type, PgInt4Type, PgInt8Type, PgFloat4Type, PgFloat8Type, PgNumericType, PgDateType,
              PgTimestampType, PgIntervalType, PgTextType, PgVarcharType, PgBpcharType>(
            [](auto typed) { return typed.getNullability(); })
        .Default([](auto) { return PgNullability::Never; });
}

mlir::Type withPgNullability(mlir::Type type, PgNullability nullability) {
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
        .Case<PgBoolType>([&](auto typed) { return PgBoolType::get(typed.getContext(), nullability); })
        .Case<PgInt2Type>([&](auto typed) { return PgInt2Type::get(typed.getContext(), nullability); })
        .Case<PgInt4Type>([&](auto typed) { return PgInt4Type::get(typed.getContext(), nullability); })
        .Case<PgInt8Type>([&](auto typed) { return PgInt8Type::get(typed.getContext(), nullability); })
        .Case<PgFloat4Type>([&](auto typed) { return PgFloat4Type::get(typed.getContext(), nullability); })
        .Case<PgFloat8Type>([&](auto typed) { return PgFloat8Type::get(typed.getContext(), nullability); })
        .Case<PgNumericType>(
            [&](auto typed) { return PgNumericType::get(typed.getContext(), typed.getTypmod(), nullability); })
        .Case<PgDateType>([&](auto typed) { return PgDateType::get(typed.getContext(), nullability); })
        .Case<PgTimestampType>(
            [&](auto typed) { return PgTimestampType::get(typed.getContext(), typed.getTypmod(), nullability); })
        .Case<PgIntervalType>(
            [&](auto typed) { return PgIntervalType::get(typed.getContext(), typed.getTypmod(), nullability); })
        .Case<PgTextType>(
            [&](auto typed) { return PgTextType::get(typed.getContext(), typed.getCollation(), nullability); })
        .Case<PgVarcharType>([&](auto typed) {
            return PgVarcharType::get(typed.getContext(), typed.getTypmod(), typed.getCollation(), nullability);
        })
        .Case<PgBpcharType>([&](auto typed) {
            return PgBpcharType::get(typed.getContext(), typed.getTypmod(), typed.getCollation(), nullability);
        })
        .Default([](auto) { return mlir::Type(); });
}

mlir::Type getPgPhysicalCarrierType(mlir::Type type) {
    auto* context = type.getContext();
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
        .Case<PgBoolType>([&](auto) { return mlir::IntegerType::get(context, 1); })
        .Case<PgInt2Type>([&](auto) { return mlir::IntegerType::get(context, 16); })
        .Case<PgInt4Type>([&](auto) { return mlir::IntegerType::get(context, 32); })
        .Case<PgInt8Type>([&](auto) { return mlir::IntegerType::get(context, 64); })
        .Case<PgFloat4Type>([&](auto) { return mlir::Float32Type::get(context); })
        .Case<PgFloat8Type>([&](auto) { return mlir::Float64Type::get(context); })
        .Case<PgNumericType, PgTimestampType>([&](auto) { return mlir::IntegerType::get(context, 64); })
        .Case<PgDateType>([&](auto) { return mlir::IntegerType::get(context, 32); })
        .Case<PgIntervalType>([&](auto) {
            return mlir::TupleType::get(context,
                                        {mlir::IntegerType::get(context, 64), mlir::IntegerType::get(context, 32),
                                         mlir::IntegerType::get(context, 32)});
        })
        .Case<PgTextType, PgVarcharType, PgBpcharType>([&](auto) { return mlir::util::VarLen32Type::get(context); })
        .Default([](auto) { return mlir::Type(); });
}

#define DEFINE_PG_TYPE_OID_METHODS(TYPE, OID)                                                                          \
    PgOid TYPE::getPgTypeOid() const {                                                                                 \
        return OID;                                                                                                    \
    }                                                                                                                  \
    mlir::Type TYPE::getPhysicalCarrierType() const {                                                                  \
        return getPgPhysicalCarrierType(*this);                                                                        \
    }

#define DEFINE_PG_NO_METADATA_METHODS(TYPE, OID)                                                                       \
    DEFINE_PG_TYPE_OID_METHODS(TYPE, OID)                                                                              \
    int32_t TYPE::getTypmod() const {                                                                                  \
        return kPgTypmodUnconstrained;                                                                                 \
    }                                                                                                                  \
    PgOid TYPE::getCollation() const {                                                                                 \
        return InvalidOid;                                                                                             \
    }

DEFINE_PG_NO_METADATA_METHODS(PgBoolType, BOOLOID)
DEFINE_PG_NO_METADATA_METHODS(PgInt2Type, INT2OID)
DEFINE_PG_NO_METADATA_METHODS(PgInt4Type, INT4OID)
DEFINE_PG_NO_METADATA_METHODS(PgInt8Type, INT8OID)
DEFINE_PG_NO_METADATA_METHODS(PgFloat4Type, FLOAT4OID)
DEFINE_PG_NO_METADATA_METHODS(PgFloat8Type, FLOAT8OID)
DEFINE_PG_NO_METADATA_METHODS(PgDateType, DATEOID)

DEFINE_PG_TYPE_OID_METHODS(PgNumericType, NUMERICOID)
PgOid PgNumericType::getCollation() const {
    return InvalidOid;
}
DEFINE_PG_TYPE_OID_METHODS(PgTimestampType, TIMESTAMPOID)
PgOid PgTimestampType::getCollation() const {
    return InvalidOid;
}
DEFINE_PG_TYPE_OID_METHODS(PgIntervalType, INTERVALOID)
PgOid PgIntervalType::getCollation() const {
    return InvalidOid;
}
DEFINE_PG_TYPE_OID_METHODS(PgTextType, TEXTOID)
int32_t PgTextType::getTypmod() const {
    return kPgTypmodUnconstrained;
}
DEFINE_PG_TYPE_OID_METHODS(PgVarcharType, VARCHAROID)
DEFINE_PG_TYPE_OID_METHODS(PgBpcharType, BPCHAROID)

#undef DEFINE_PG_NO_METADATA_METHODS
#undef DEFINE_PG_TYPE_OID_METHODS

void DBDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
        >();
}

} // namespace mlir::db
