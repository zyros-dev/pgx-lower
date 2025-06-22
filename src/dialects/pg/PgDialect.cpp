#include "dialects/pg/PgDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::pg;

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

void PgDialect::initialize() {
    addOperations<ScanTableOp, ReadTupleOp, GetIntFieldOp, GetTextFieldOp>();
    addTypes<
        // String types
        TextType, CharType, VarCharType,
        // Integer types  
        PgSmallIntType, PgIntegerType, PgBigIntType,
        // Decimal/float types
        NumericType, RealType, DoubleType, MoneyType,
        // Boolean and binary types
        BooleanType, ByteaType,
        // Date/time types
        DateType, TimeType, TimeTzType, TimestampType, TimestampTzType, IntervalType,
        // Network types
        UuidType, InetType, CidrType, MacAddrType,
        // Bit types
        BitType, VarBitType,
        // System types
        PgTupleType, TableHandleType, TupleHandleType
    >();
}

PgDialect::PgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PgDialect>()) {
    initialize();
}

Type PgDialect::parseType(DialectAsmParser &parser) const {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return Type();
    
    // String types
    if (keyword == "text")
        return TextType::get(getContext());
    if (keyword == "char")
        return CharType::get(getContext(), 1); // Default length
    if (keyword == "varchar")
        return VarCharType::get(getContext(), 255); // Default max length
    
    // Integer types
    if (keyword == "smallint")
        return PgSmallIntType::get(getContext());
    if (keyword == "integer")
        return PgIntegerType::get(getContext());
    if (keyword == "bigint")
        return PgBigIntType::get(getContext());
    
    // Decimal/float types  
    if (keyword == "numeric")
        return NumericType::get(getContext());
    if (keyword == "real")
        return RealType::get(getContext());
    if (keyword == "double")
        return DoubleType::get(getContext());
    if (keyword == "money")
        return MoneyType::get(getContext());
    
    // Boolean and binary types
    if (keyword == "boolean")
        return BooleanType::get(getContext());
    if (keyword == "bytea")
        return ByteaType::get(getContext());
    
    // Date/time types
    if (keyword == "date")
        return DateType::get(getContext());
    if (keyword == "time")
        return TimeType::get(getContext());
    if (keyword == "timetz")
        return TimeTzType::get(getContext());
    if (keyword == "timestamp")
        return TimestampType::get(getContext());
    if (keyword == "timestamptz")
        return TimestampTzType::get(getContext());
    if (keyword == "interval")
        return IntervalType::get(getContext());
    
    // Network types
    if (keyword == "uuid")
        return UuidType::get(getContext());
    if (keyword == "inet")
        return InetType::get(getContext());
    if (keyword == "cidr")
        return CidrType::get(getContext());
    if (keyword == "macaddr")
        return MacAddrType::get(getContext());
    
    // Bit types
    if (keyword == "bit")
        return BitType::get(getContext(), 1); // Default length
    if (keyword == "varbit")
        return VarBitType::get(getContext(), 8); // Default max length
    
    // System types
    if (keyword == "table_handle")
        return TableHandleType::get(getContext());
    if (keyword == "tuple_handle")
        return TupleHandleType::get(getContext());
    
    return Type();
}

void PgDialect::printType(Type type, DialectAsmPrinter &printer) const {
    // String types
    if (auto textType = mlir::dyn_cast<TextType>(type)) {
        printer << "text";
        return;
    }
    if (auto charType = mlir::dyn_cast<CharType>(type)) {
        printer << "char";
        return;
    }
    if (auto varcharType = mlir::dyn_cast<VarCharType>(type)) {
        printer << "varchar";
        return;
    }
    
    // Integer types
    if (auto smallintType = mlir::dyn_cast<PgSmallIntType>(type)) {
        printer << "smallint";
        return;
    }
    if (auto integerType = mlir::dyn_cast<PgIntegerType>(type)) {
        printer << "integer";
        return;
    }
    if (auto bigintType = mlir::dyn_cast<PgBigIntType>(type)) {
        printer << "bigint";
        return;
    }
    
    // Decimal/float types
    if (auto numericType = mlir::dyn_cast<NumericType>(type)) {
        printer << "numeric";
        return;
    }
    if (auto realType = mlir::dyn_cast<RealType>(type)) {
        printer << "real";
        return;
    }
    if (auto doubleType = mlir::dyn_cast<DoubleType>(type)) {
        printer << "double";
        return;
    }
    if (auto moneyType = mlir::dyn_cast<MoneyType>(type)) {
        printer << "money";
        return;
    }
    
    // Boolean and binary types
    if (auto booleanType = mlir::dyn_cast<BooleanType>(type)) {
        printer << "boolean";
        return;
    }
    if (auto byteaType = mlir::dyn_cast<ByteaType>(type)) {
        printer << "bytea";
        return;
    }
    
    // Date/time types
    if (auto dateType = mlir::dyn_cast<DateType>(type)) {
        printer << "date";
        return;
    }
    if (auto timeType = mlir::dyn_cast<TimeType>(type)) {
        printer << "time";
        return;
    }
    if (auto timetzType = mlir::dyn_cast<TimeTzType>(type)) {
        printer << "timetz";
        return;
    }
    if (auto timestampType = mlir::dyn_cast<TimestampType>(type)) {
        printer << "timestamp";
        return;
    }
    if (auto timestamptzType = mlir::dyn_cast<TimestampTzType>(type)) {
        printer << "timestamptz";
        return;
    }
    if (auto intervalType = mlir::dyn_cast<IntervalType>(type)) {
        printer << "interval";
        return;
    }
    
    // Network types
    if (auto uuidType = mlir::dyn_cast<UuidType>(type)) {
        printer << "uuid";
        return;
    }
    if (auto inetType = mlir::dyn_cast<InetType>(type)) {
        printer << "inet";
        return;
    }
    if (auto cidrType = mlir::dyn_cast<CidrType>(type)) {
        printer << "cidr";
        return;
    }
    if (auto macaddrType = mlir::dyn_cast<MacAddrType>(type)) {
        printer << "macaddr";
        return;
    }
    
    // Bit types
    if (auto bitType = mlir::dyn_cast<BitType>(type)) {
        printer << "bit";
        return;
    }
    if (auto varbitType = mlir::dyn_cast<VarBitType>(type)) {
        printer << "varbit";
        return;
    }
    
    // System types
    if (auto tableType = mlir::dyn_cast<TableHandleType>(type)) {
        printer << "table_handle";
        return;
    }
    if (auto tupleType = mlir::dyn_cast<TupleHandleType>(type)) {
        printer << "tuple_handle";
        return;
    }
    
    printer << "<<unknown pg type>>";
}

//===----------------------------------------------------------------------===//
// Type Storage Implementations
//===----------------------------------------------------------------------===//

namespace mlir::pg::detail {

//===----------------------------------------------------------------------===//
// String Type Storages
//===----------------------------------------------------------------------===//

/// Storage for TextType
struct TextTypeStorage : public TypeStorage {
    TextTypeStorage() {}
    
    using KeyTy = std::tuple<>; // Empty tuple for parameterless types
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static TextTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<TextTypeStorage>()) TextTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

/// Storage for CharType
struct CharTypeStorage : public TypeStorage {
    unsigned length;
    
    CharTypeStorage(unsigned length) : length(length) {}
    
    using KeyTy = unsigned;
    
    bool operator==(const KeyTy &key) const {
        return key == length;
    }
    
    static CharTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<CharTypeStorage>()) CharTypeStorage(key);
    }
};

/// Storage for VarCharType
struct VarCharTypeStorage : public TypeStorage {
    unsigned maxLength;
    
    VarCharTypeStorage(unsigned maxLength) : maxLength(maxLength) {}
    
    using KeyTy = unsigned;
    
    bool operator==(const KeyTy &key) const {
        return key == maxLength;
    }
    
    static VarCharTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<VarCharTypeStorage>()) VarCharTypeStorage(key);
    }
};

//===----------------------------------------------------------------------===//
// Integer Type Storages
//===----------------------------------------------------------------------===//

/// Storage for SmallIntType
struct SmallIntTypeStorage : public TypeStorage {
    SmallIntTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static SmallIntTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<SmallIntTypeStorage>()) SmallIntTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

/// Storage for IntegerType
struct IntegerTypeStorage : public TypeStorage {
    IntegerTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static IntegerTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<IntegerTypeStorage>()) IntegerTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

/// Storage for BigIntType
struct BigIntTypeStorage : public TypeStorage {
    BigIntTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static BigIntTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<BigIntTypeStorage>()) BigIntTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

/// Storage for NumericType
struct NumericTypeStorage : public TypeStorage {
    unsigned precision;
    unsigned scale;
    
    NumericTypeStorage(unsigned precision, unsigned scale)
        : precision(precision), scale(scale) {}
    
    using KeyTy = std::pair<unsigned, unsigned>;
    
    bool operator==(const KeyTy &key) const {
        return key.first == precision && key.second == scale;
    }
    
    static NumericTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<NumericTypeStorage>()) 
            NumericTypeStorage(key.first, key.second);
    }
};

//===----------------------------------------------------------------------===//
// Template for simple parameterless types
//===----------------------------------------------------------------------===//

template<typename T>
struct SimpleTypeStorage : public TypeStorage {
    SimpleTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static T *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<T>()) T();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

// Decimal/float type storages
struct RealTypeStorage : public SimpleTypeStorage<RealTypeStorage> {};
struct DoubleTypeStorage : public SimpleTypeStorage<DoubleTypeStorage> {};
struct MoneyTypeStorage : public SimpleTypeStorage<MoneyTypeStorage> {};

// Boolean and binary type storages
struct BooleanTypeStorage : public SimpleTypeStorage<BooleanTypeStorage> {};
struct ByteaTypeStorage : public SimpleTypeStorage<ByteaTypeStorage> {};

// Date/time type storages
struct TimeTypeStorage : public SimpleTypeStorage<TimeTypeStorage> {};
struct TimeTzTypeStorage : public SimpleTypeStorage<TimeTzTypeStorage> {};
struct TimestampTypeStorage : public SimpleTypeStorage<TimestampTypeStorage> {};
struct TimestampTzTypeStorage : public SimpleTypeStorage<TimestampTzTypeStorage> {};
struct IntervalTypeStorage : public SimpleTypeStorage<IntervalTypeStorage> {};

// Network type storages
struct UuidTypeStorage : public SimpleTypeStorage<UuidTypeStorage> {};
struct InetTypeStorage : public SimpleTypeStorage<InetTypeStorage> {};
struct CidrTypeStorage : public SimpleTypeStorage<CidrTypeStorage> {};
struct MacAddrTypeStorage : public SimpleTypeStorage<MacAddrTypeStorage> {};

/// Storage for BitType
struct BitTypeStorage : public TypeStorage {
    unsigned length;
    
    BitTypeStorage(unsigned length) : length(length) {}
    
    using KeyTy = unsigned;
    
    bool operator==(const KeyTy &key) const {
        return key == length;
    }
    
    static BitTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<BitTypeStorage>()) BitTypeStorage(key);
    }
};

/// Storage for VarBitType
struct VarBitTypeStorage : public TypeStorage {
    unsigned maxLength;
    
    VarBitTypeStorage(unsigned maxLength) : maxLength(maxLength) {}
    
    using KeyTy = unsigned;
    
    bool operator==(const KeyTy &key) const {
        return key == maxLength;
    }
    
    static VarBitTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<VarBitTypeStorage>()) VarBitTypeStorage(key);
    }
};

/// Storage for DateType
struct DateTypeStorage : public TypeStorage {
    DateTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static DateTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<DateTypeStorage>()) DateTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

/// Storage for PgTupleType
struct PgTupleTypeStorage : public TypeStorage {
    ArrayRef<Type> fieldTypes;
    
    PgTupleTypeStorage(ArrayRef<Type> fieldTypes) : fieldTypes(fieldTypes) {}
    
    using KeyTy = ArrayRef<Type>;
    
    bool operator==(const KeyTy &key) const {
        return key == fieldTypes;
    }
    
    static PgTupleTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        ArrayRef<Type> types = allocator.copyInto(key);
        return new (allocator.allocate<PgTupleTypeStorage>()) PgTupleTypeStorage(types);
    }
};

/// Storage for TableHandleType
struct TableHandleTypeStorage : public TypeStorage {
    TableHandleTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static TableHandleTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<TableHandleTypeStorage>()) TableHandleTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

/// Storage for TupleHandleType
struct TupleHandleTypeStorage : public TypeStorage {
    TupleHandleTypeStorage() {}
    
    using KeyTy = std::tuple<>;
    
    bool operator==(const KeyTy &key) const { return true; }
    
    static TupleHandleTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<TupleHandleTypeStorage>()) TupleHandleTypeStorage();
    }
    
    static KeyTy getKey() {
        return std::make_tuple();
    }
};

} // namespace mlir::pg::detail

//===----------------------------------------------------------------------===//
// Type Implementations
//===----------------------------------------------------------------------===//

TextType TextType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

NumericType NumericType::get(MLIRContext *context, unsigned precision, unsigned scale) {
    return Base::get(context, std::make_pair(precision, scale));
}

unsigned NumericType::getPrecision() const {
    return getImpl()->precision;
}

unsigned NumericType::getScale() const {
    return getImpl()->scale;
}

DateType DateType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// String Type Implementations  
//===----------------------------------------------------------------------===//

CharType CharType::get(MLIRContext *context, unsigned length) {
    return Base::get(context, length);
}

unsigned CharType::getLength() const {
    return getImpl()->length;
}

VarCharType VarCharType::get(MLIRContext *context, unsigned maxLength) {
    return Base::get(context, maxLength);
}

unsigned VarCharType::getMaxLength() const {
    return getImpl()->maxLength;
}

//===----------------------------------------------------------------------===//
// Integer Type Implementations
//===----------------------------------------------------------------------===//

PgSmallIntType PgSmallIntType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

PgIntegerType PgIntegerType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

PgBigIntType PgBigIntType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// Decimal/Float Type Implementations
//===----------------------------------------------------------------------===//

RealType RealType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

DoubleType DoubleType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

MoneyType MoneyType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// Boolean and Binary Type Implementations
//===----------------------------------------------------------------------===//

BooleanType BooleanType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

ByteaType ByteaType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// Date/Time Type Implementations
//===----------------------------------------------------------------------===//

TimeType TimeType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

TimeTzType TimeTzType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

TimestampType TimestampType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

TimestampTzType TimestampTzType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

IntervalType IntervalType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// Network Type Implementations
//===----------------------------------------------------------------------===//

UuidType UuidType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

InetType InetType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

CidrType CidrType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

MacAddrType MacAddrType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// Bit Type Implementations
//===----------------------------------------------------------------------===//

BitType BitType::get(MLIRContext *context, unsigned length) {
    return Base::get(context, length);
}

unsigned BitType::getLength() const {
    return getImpl()->length;
}

VarBitType VarBitType::get(MLIRContext *context, unsigned maxLength) {
    return Base::get(context, maxLength);
}

unsigned VarBitType::getMaxLength() const {
    return getImpl()->maxLength;
}

PgTupleType PgTupleType::get(MLIRContext *context, ArrayRef<Type> fieldTypes) {
    return Base::get(context, fieldTypes);
}

ArrayRef<Type> PgTupleType::getFieldTypes() const {
    return getImpl()->fieldTypes;
}

unsigned PgTupleType::getNumFields() const {
    return getFieldTypes().size();
}

Type PgTupleType::getFieldType(unsigned index) const {
    assert(index < getNumFields() && "Field index out of bounds");
    return getFieldTypes()[index];
}

TableHandleType TableHandleType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

TupleHandleType TupleHandleType::get(MLIRContext *context) {
    return Base::get(context, std::make_tuple());
}

//===----------------------------------------------------------------------===//
// Operation Implementations
//===----------------------------------------------------------------------===//

void ScanTableOp::build(OpBuilder &builder, OperationState &result, 
                        StringRef tableName) {
    result.addAttribute("table_name", builder.getStringAttr(tableName));
    result.addTypes(TableHandleType::get(builder.getContext()));
}

ParseResult ScanTableOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr tableName;
    Type resultType;
    
    if (parser.parseAttribute(tableName, "table_name", result.attributes) ||
        parser.parseColonType(resultType))
        return failure();
    
    result.addTypes(resultType);
    return success();
}

void ScanTableOp::print(OpAsmPrinter &printer) {
    printer << " ";
    printer.printAttributeWithoutType((*this)->getAttr("table_name"));
    printer << " : ";
    printer.printType(getResult().getType());
}

LogicalResult ScanTableOp::verify() {
    if (!(*this)->getAttr("table_name"))
        return emitOpError("missing table_name attribute");
    
    if (!mlir::isa<TableHandleType>(getResult().getType()))
        return emitOpError("result must be !pg.table_handle type");
    
    return success();
}

void ReadTupleOp::build(OpBuilder &builder, OperationState &result, 
                        Value tableHandle, ArrayRef<Type> fieldTypes) {
    result.addOperands(tableHandle);
    result.addTypes(TupleHandleType::get(builder.getContext()));
}

ParseResult ReadTupleOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand tableHandle;
    Type tupleType;
    
    if (parser.parseOperand(tableHandle) ||
        parser.parseColonType(tupleType) ||
        parser.resolveOperand(tableHandle, TableHandleType::get(parser.getContext()), result.operands))
        return failure();
    
    result.addTypes(tupleType);
    return success();
}

void ReadTupleOp::print(OpAsmPrinter &printer) {
    printer << " ";
    printer.printOperand(getTableHandle());
    printer << " : ";
    printer.printType(getResult().getType());
}

LogicalResult ReadTupleOp::verify() {
    if (!mlir::isa<TableHandleType>(getTableHandle().getType()))
        return emitOpError("operand must be !pg.table_handle type");
    
    if (!mlir::isa<TupleHandleType>(getResult().getType()))
        return emitOpError("result must be !pg.tuple_handle type");
    
    return success();
}

void GetIntFieldOp::build(OpBuilder &builder, OperationState &result, 
                         Value tuple, unsigned fieldIndex) {
    result.addOperands(tuple);
    result.addAttribute("field_index", builder.getI32IntegerAttr(fieldIndex));
    result.addTypes({builder.getI32Type(), builder.getI1Type()}); // value, null_flag
}

ParseResult GetIntFieldOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand tuple;
    IntegerAttr fieldIndex;
    Type tupleType, resultType, nullType;
    
    if (parser.parseOperand(tuple) ||
        parser.parseComma() ||
        parser.parseAttribute(fieldIndex, "field_index", result.attributes) ||
        parser.parseColonType(tupleType) ||
        parser.parseArrow() ||
        parser.parseLParen() ||
        parser.parseType(resultType) ||
        parser.parseComma() ||
        parser.parseType(nullType) ||
        parser.parseRParen() ||
        parser.resolveOperand(tuple, tupleType, result.operands))
        return failure();
    
    result.addTypes({resultType, nullType});
    return success();
}

void GetIntFieldOp::print(OpAsmPrinter &printer) {
    printer << " ";
    printer.printOperand(getTuple());
    printer << ", ";
    printer.printAttributeWithoutType((*this)->getAttr("field_index"));
    printer << " : ";
    printer.printType(getTuple().getType());
    printer << " -> (";
    printer.printType(getResult().getType());
    printer << ", ";
    printer.printType(getNullFlag().getType());
    printer << ")";
}

LogicalResult GetIntFieldOp::verify() {
    if (!mlir::isa<TupleHandleType>(getTuple().getType()))
        return emitOpError("operand must be !pg.tuple_handle type");
    
    if (!getResult().getType().isInteger(32))
        return emitOpError("result must be i32 type");
    
    if (!getNullFlag().getType().isInteger(1))
        return emitOpError("null flag must be i1 type");
    
    return success();
}

void GetTextFieldOp::build(OpBuilder &builder, OperationState &result, 
                          Value tuple, unsigned fieldIndex) {
    result.addOperands(tuple);
    result.addAttribute("field_index", builder.getI32IntegerAttr(fieldIndex));
    result.addTypes({builder.getI64Type(), builder.getI1Type()}); // pointer, null_flag
}

ParseResult GetTextFieldOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand tuple;
    IntegerAttr fieldIndex;
    Type tupleType, resultType, nullType;
    
    if (parser.parseOperand(tuple) ||
        parser.parseComma() ||
        parser.parseAttribute(fieldIndex, "field_index", result.attributes) ||
        parser.parseColonType(tupleType) ||
        parser.parseArrow() ||
        parser.parseLParen() ||
        parser.parseType(resultType) ||
        parser.parseComma() ||
        parser.parseType(nullType) ||
        parser.parseRParen() ||
        parser.resolveOperand(tuple, tupleType, result.operands))
        return failure();
    
    result.addTypes({resultType, nullType});
    return success();
}

void GetTextFieldOp::print(OpAsmPrinter &printer) {
    printer << " ";
    printer.printOperand(getTuple());
    printer << ", ";
    printer.printAttributeWithoutType((*this)->getAttr("field_index"));
    printer << " : ";
    printer.printType(getTuple().getType());
    printer << " -> (";
    printer.printType(getResult().getType());
    printer << ", ";
    printer.printType(getNullFlag().getType());
    printer << ")";
}

LogicalResult GetTextFieldOp::verify() {
    if (!mlir::isa<TupleHandleType>(getTuple().getType()))
        return emitOpError("operand must be !pg.tuple_handle type");
    
    if (!getResult().getType().isInteger(64))
        return emitOpError("result must be i64 type (text pointer)");
    
    if (!getNullFlag().getType().isInteger(1))
        return emitOpError("null flag must be i1 type");
    
    return success();
}

// TODO: Include auto-generated definitions when TableGen is fixed
// #define GET_OP_CLASSES
// #include "dialects/pg/PgOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "dialects/pg/PgTypes.cpp.inc"