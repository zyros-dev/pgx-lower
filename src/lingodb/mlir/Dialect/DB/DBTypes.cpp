#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "lingodb/mlir/Dialect/util/UtilTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/STLExtras.h>

#include <cstdint>
#include <limits>

namespace mlir { namespace db {

namespace {
constexpr int32_t kPgTypmodUnconstrained = -1;
constexpr PgOid InvalidOid = 0;
constexpr PgOid BOOLOID = 16;
constexpr PgOid INT8OID = 20;
constexpr PgOid INT2OID = 21;
constexpr PgOid INT4OID = 23;
constexpr PgOid TEXTOID = 25;
constexpr PgOid FLOAT4OID = 700;
constexpr PgOid FLOAT8OID = 701;
constexpr PgOid BPCHAROID = 1042;
constexpr PgOid VARCHAROID = 1043;
constexpr PgOid DATEOID = 1082;
constexpr PgOid TIMESTAMPOID = 1114;
constexpr PgOid INTERVALOID = 1186;
constexpr PgOid NUMERICOID = 1700;

auto parseKeywordEqual(AsmParser& parser, llvm::StringRef keyword) -> ParseResult {
    if (parser.parseKeyword(keyword) || parser.parseEqual()) {
        return failure();
    }
    return success();
}

auto parseUInt32Value(AsmParser& parser) -> FailureOr<uint32_t> {
    uint64_t parsed{};
    if (parser.parseInteger(parsed)) {
        return failure();
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        return parser.emitError(parser.getCurrentLocation(), "expected 32-bit unsigned integer");
    }
    return static_cast<uint32_t>(parsed);
}

auto parseInt16Value(AsmParser& parser) -> FailureOr<int16_t> {
    int64_t parsed{};
    if (parser.parseInteger(parsed)) {
        return failure();
    }
    if (parsed < std::numeric_limits<int16_t>::min() || parsed > std::numeric_limits<int16_t>::max()) {
        return parser.emitError(parser.getCurrentLocation(), "expected 16-bit signed integer");
    }
    return static_cast<int16_t>(parsed);
}

auto parseInt32Value(AsmParser& parser) -> FailureOr<int32_t> {
    int64_t parsed{};
    if (parser.parseInteger(parsed)) {
        return failure();
    }
    if (parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) {
        return parser.emitError(parser.getCurrentLocation(), "expected 32-bit signed integer");
    }
    return static_cast<int32_t>(parsed);
}

auto parseBoolValue(AsmParser& parser) -> FailureOr<bool> {
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
        return failure();
    }
    if (keyword == "true") {
        return true;
    }
    if (keyword == "false") {
        return false;
    }
    return parser.emitError(parser.getCurrentLocation(), "expected true or false");
}

auto parsePgNullabilityValue(AsmParser& parser) -> FailureOr<PgNullability> {
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
        return failure();
    }
    if (keyword == "never") {
        return PgNullability::Never;
    }
    if (keyword == "maybe") {
        return PgNullability::Maybe;
    }
    return parser.emitError(parser.getCurrentLocation(), "expected never or maybe");
}

auto parsePgRowFieldOriginValue(AsmParser& parser) -> FailureOr<PgRowFieldOrigin> {
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
        return failure();
    }
    if (keyword == "base") {
        return PgRowFieldOrigin::base;
    }
    if (keyword == "computed") {
        return PgRowFieldOrigin::computed;
    }
    if (keyword == "aggregate") {
        return PgRowFieldOrigin::aggregate;
    }
    if (keyword == "join") {
        return PgRowFieldOrigin::join;
    }
    if (keyword == "subquery") {
        return PgRowFieldOrigin::subquery;
    }
    if (keyword == "unknown") {
        return PgRowFieldOrigin::unknown;
    }
    return parser.emitError(parser.getCurrentLocation(), "expected row field origin");
}

void printPgRowNullability(AsmPrinter& printer, PgNullability nullability) {
    switch (nullability) {
    case PgNullability::Never: printer << "never"; break;
    case PgNullability::Maybe: printer << "maybe"; break;
    }
}

void printPgRowFieldOrigin(AsmPrinter& printer, PgRowFieldOrigin origin) {
    switch (origin) {
    case PgRowFieldOrigin::base: printer << "base"; break;
    case PgRowFieldOrigin::computed: printer << "computed"; break;
    case PgRowFieldOrigin::aggregate: printer << "aggregate"; break;
    case PgRowFieldOrigin::join: printer << "join"; break;
    case PgRowFieldOrigin::subquery: printer << "subquery"; break;
    case PgRowFieldOrigin::unknown: printer << "unknown"; break;
    }
}

auto verifyPgRowSchemaFields(llvm::function_ref<InFlightDiagnostic()> emitError, llvm::ArrayRef<PgRowFieldAttr> fields)
    -> LogicalResult {
    for (auto [position, field] : llvm::enumerate(fields)) {
        if (!field) {
            return emitError() << "row schema requires non-null row field attributes";
        }
        if (field.getIndex() != position) {
            return emitError() << "row field index must equal ordered position and be contiguous from zero";
        }
    }
    return success();
}

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

auto parsePgRowSchemaTypeParameter(AsmParser& parser) -> PgRowSchemaAttr {
    if (parser.parseLess()) {
        return {};
    }
    Attribute attr;
    if (parser.parseAttribute(attr)) {
        return {};
    }
    auto schema = mlir::dyn_cast<PgRowSchemaAttr>(attr);
    if (!schema) {
        parser.emitError(parser.getCurrentLocation(), "expected pg_row_schema attribute");
        return {};
    }
    if (parser.parseGreater()) {
        return {};
    }
    return schema;
}

template<typename TypeT>
auto parsePgRowContainerType(AsmParser& parser) -> Type {
    auto schema = parsePgRowSchemaTypeParameter(parser);
    if (!schema) {
        return {};
    }
    return TypeT::get(parser.getContext(), schema);
}

template<typename TypeT>
void printPgRowContainerType(TypeT type, AsmPrinter& printer) {
    printer << "<";
    printer.printAttribute(type.getSchema());
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

Type PgRowType::parse(AsmParser& parser) {
    return parsePgRowContainerType<PgRowType>(parser);
}

void PgRowType::print(AsmPrinter& printer) const {
    printPgRowContainerType(*this, printer);
}

Type PgRowStreamType::parse(AsmParser& parser) {
    return parsePgRowContainerType<PgRowStreamType>(parser);
}

void PgRowStreamType::print(AsmPrinter& printer) const {
    printPgRowContainerType(*this, printer);
}

Attribute PgRowFieldAttr::parse(AsmParser& parser, Type odsType) {
    auto loc = parser.getCurrentLocation();
    if (parser.parseLess()) {
        return {};
    }

    if (failed(parseKeywordEqual(parser, "index"))) {
        return {};
    }
    auto index = parseUInt32Value(parser);
    if (failed(index) || parser.parseComma() || failed(parseKeywordEqual(parser, "relid"))) {
        return {};
    }
    auto relid = parsePgOidValue(parser);
    if (failed(relid) || parser.parseComma() || failed(parseKeywordEqual(parser, "varno"))) {
        return {};
    }
    auto varno = parseUInt32Value(parser);
    if (failed(varno) || parser.parseComma() || failed(parseKeywordEqual(parser, "attno"))) {
        return {};
    }
    auto attno = parseInt16Value(parser);
    if (failed(attno) || parser.parseComma() || failed(parseKeywordEqual(parser, "name"))) {
        return {};
    }
    StringAttr name;
    if (parser.parseAttribute(name) || parser.parseComma() || failed(parseKeywordEqual(parser, "type"))) {
        return {};
    }
    Type type;
    if (parser.parseType(type) || parser.parseComma() || failed(parseKeywordEqual(parser, "oid"))) {
        return {};
    }
    auto oid = parsePgOidValue(parser);
    if (failed(oid) || parser.parseComma() || failed(parseKeywordEqual(parser, "typmod"))) {
        return {};
    }
    auto typmod = parseInt32Value(parser);
    if (failed(typmod) || parser.parseComma() || failed(parseKeywordEqual(parser, "collation"))) {
        return {};
    }
    auto collation = parsePgOidValue(parser);
    if (failed(collation) || parser.parseComma() || failed(parseKeywordEqual(parser, "nullable"))) {
        return {};
    }
    auto nullability = parsePgNullabilityValue(parser);
    if (failed(nullability) || parser.parseComma() || failed(parseKeywordEqual(parser, "resjunk"))) {
        return {};
    }
    auto resjunk = parseBoolValue(parser);
    if (failed(resjunk) || parser.parseComma() || failed(parseKeywordEqual(parser, "origin"))) {
        return {};
    }
    auto origin = parsePgRowFieldOriginValue(parser);
    if (failed(origin) || parser.parseGreater()) {
        return {};
    }

    return PgRowFieldAttr::getChecked([&]() { return parser.emitError(loc); }, parser.getContext(), *index, *relid,
                                      *varno, *attno, name, type, *oid, *typmod, *collation, *nullability, *resjunk,
                                      *origin);
}

void PgRowFieldAttr::print(AsmPrinter& printer) const {
    printer << "<index = " << getIndex() << ", relid = " << getRelid() << ", varno = " << getVarno()
            << ", attno = " << getAttno() << ", name = ";
    printer.printAttribute(getName());
    printer << ", type = ";
    printer.printStrippedAttrOrType(getType());
    printer << ", oid = " << getOid() << ", typmod = " << getTypmod() << ", collation = " << getCollation()
            << ", nullable = ";
    printPgRowNullability(printer, getNullability());
    printer << ", resjunk = " << (getResjunk() ? "true" : "false") << ", origin = ";
    printPgRowFieldOrigin(printer, getOrigin());
    printer << ">";
}

Attribute PgRowSchemaAttr::parse(AsmParser& parser, Type odsType) {
    auto loc = parser.getCurrentLocation();
    if (parser.parseLess() || parser.parseLSquare()) {
        return {};
    }

    SmallVector<PgRowFieldAttr> fields;
    if (failed(parser.parseOptionalRSquare())) {
        do {
            Attribute attr;
            if (parser.parseAttribute(attr)) {
                return {};
            }
            auto field = mlir::dyn_cast<PgRowFieldAttr>(attr);
            if (!field) {
                parser.emitError(parser.getCurrentLocation(), "expected pg_row_field attribute");
                return {};
            }
            fields.push_back(field);
        } while (succeeded(parser.parseOptionalComma()));

        if (parser.parseRSquare()) {
            return {};
        }
    }

    if (parser.parseGreater()) {
        return {};
    }
    if (failed(verifyPgRowSchemaFields([&]() { return parser.emitError(loc); }, fields))) {
        return {};
    }
    llvm::ArrayRef<PgRowFieldAttr> fieldRef(fields);
    return PgRowSchemaAttr::getChecked([&]() { return parser.emitError(loc); }, parser.getContext(), fieldRef);
}

void PgRowSchemaAttr::print(AsmPrinter& printer) const {
    printer << "<[";
    llvm::interleaveComma(getFields(), printer, [&](PgRowFieldAttr field) { printer.printAttribute(field); });
    printer << "]>";
}

LogicalResult PgRowFieldAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError, uint32_t index, PgOid relid,
                                     uint32_t varno, int16_t attno, StringAttr name, Type type, PgOid oid, int32_t typmod,
                                     PgOid collation, PgNullability nullability, bool resjunk, PgRowFieldOrigin origin) {
    if (!name) {
        return emitError() << "row field requires a name";
    }
    if (!type || !mlir::db::isPgValueType(type)) {
        return emitError() << "row field type must be a PostgreSQL semantic type";
    }
    if (oid != mlir::db::getPgTypeOid(type)) {
        return emitError() << "row field oid must match its PostgreSQL semantic type";
    }
    if (typmod != mlir::db::getPgTypmod(type)) {
        return emitError() << "row field typmod must match its PostgreSQL semantic type";
    }
    if (collation != mlir::db::getPgCollation(type)) {
        return emitError() << "row field collation must match its PostgreSQL semantic type";
    }
    if (nullability != mlir::db::getPgNullability(type)) {
        return emitError() << "row field nullability must match its PostgreSQL semantic type";
    }
    (void)index;
    (void)relid;
    (void)varno;
    (void)attno;
    (void)resjunk;
    (void)origin;
    return success();
}

LogicalResult
PgRowSchemaAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError, llvm::ArrayRef<PgRowFieldAttr> fields) {
    return verifyPgRowSchemaFields(emitError, fields);
}

}} // namespace mlir::db
#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace mlir::db {

PgRowFieldLookupResult::PgRowFieldLookupResult(PgRowFieldLookupStatus status, PgRowFieldAttr field)
: status(status)
, field(field) {}

PgRowFieldLookupResult PgRowFieldLookupResult::notFound() {
    return PgRowFieldLookupResult(PgRowFieldLookupStatus::NotFound, {});
}

PgRowFieldLookupResult PgRowFieldLookupResult::found(PgRowFieldAttr field) {
    return PgRowFieldLookupResult(PgRowFieldLookupStatus::Found, field);
}

PgRowFieldLookupResult PgRowFieldLookupResult::ambiguous() {
    return PgRowFieldLookupResult(PgRowFieldLookupStatus::Ambiguous, {});
}

bool PgRowFieldLookupResult::isFound() const {
    return status == PgRowFieldLookupStatus::Found;
}

bool PgRowFieldLookupResult::isAmbiguous() const {
    return status == PgRowFieldLookupStatus::Ambiguous;
}

uint32_t PgRowFieldLookupResult::getFieldIndex() const {
    return field ? field.getIndex() : 0;
}

PgRowFieldAttr PgRowFieldLookupResult::getField() const {
    return field;
}

bool isPgRowType(mlir::Type type) {
    return mlir::isa<PgRowType>(type);
}

bool isPgRowStreamType(mlir::Type type) {
    return mlir::isa<PgRowStreamType>(type);
}

PgRowSchemaAttr getPgRowSchema(mlir::Type type) {
    if (auto rowType = mlir::dyn_cast_or_null<PgRowType>(type)) {
        return rowType.getSchema();
    }
    if (auto rowStreamType = mlir::dyn_cast_or_null<PgRowStreamType>(type)) {
        return rowStreamType.getSchema();
    }
    return {};
}

uint32_t getPgRowFieldCount(mlir::Type type) {
    if (auto schema = getPgRowSchema(type)) {
        return static_cast<uint32_t>(schema.getFields().size());
    }
    return 0;
}

PgRowFieldAttr getPgRowFieldByIndex(mlir::Type type, uint32_t index) {
    auto schema = getPgRowSchema(type);
    if (!schema || index >= schema.getFields().size()) {
        return {};
    }
    return schema.getFields()[index];
}

PgRowFieldLookupResult lookupPgRowFieldBySource(mlir::Type type, uint32_t varno, int16_t attno) {
    if (varno == 0 || attno == 0) {
        return PgRowFieldLookupResult::notFound();
    }
    auto schema = getPgRowSchema(type);
    if (!schema) {
        return PgRowFieldLookupResult::notFound();
    }

    PgRowFieldAttr found;
    for (auto field : schema.getFields()) {
        if (field.getVarno() != varno || field.getAttno() != attno) {
            continue;
        }
        if (found) {
            return PgRowFieldLookupResult::ambiguous();
        }
        found = field;
    }
    if (!found) {
        return PgRowFieldLookupResult::notFound();
    }
    return PgRowFieldLookupResult::found(found);
}

mlir::Type getPgRowFieldType(PgRowFieldAttr field) {
    return field ? field.getType() : mlir::Type();
}

PgOid getPgRowFieldOid(PgRowFieldAttr field) {
    return field ? field.getOid() : InvalidOid;
}

int32_t getPgRowFieldTypmod(PgRowFieldAttr field) {
    return field ? field.getTypmod() : kPgTypmodUnconstrained;
}

PgOid getPgRowFieldCollation(PgRowFieldAttr field) {
    return field ? field.getCollation() : InvalidOid;
}

PgNullability getPgRowFieldNullability(PgRowFieldAttr field) {
    return field ? field.getNullability() : PgNullability::Never;
}

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
