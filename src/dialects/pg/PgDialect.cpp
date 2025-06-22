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
    addTypes<TextType, NumericType, DateType, PgTupleType, TableHandleType, TupleHandleType>();
}

PgDialect::PgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PgDialect>()) {
    initialize();
}

Type PgDialect::parseType(DialectAsmParser &parser) const {
    // Manual type parser until TableGen is fixed
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return Type();
    
    if (keyword == "text")
        return TextType::get(getContext());
    if (keyword == "date")
        return DateType::get(getContext());
    if (keyword == "table_handle")
        return TableHandleType::get(getContext());
    if (keyword == "tuple_handle")
        return TupleHandleType::get(getContext());
    
    return Type();
}

void PgDialect::printType(Type type, DialectAsmPrinter &printer) const {
    // Manual type printer until TableGen is fixed
    if (auto textType = mlir::dyn_cast<TextType>(type)) {
        printer << "text";
        return;
    }
    if (auto dateType = mlir::dyn_cast<DateType>(type)) {
        printer << "date";
        return;
    }
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