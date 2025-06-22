#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pg {

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

class PgDialect : public Dialect {
public:
    explicit PgDialect(MLIRContext *context);

    static StringRef getDialectNamespace() { return "pg"; }

    // Parse type from string representation
    Type parseType(DialectAsmParser &parser) const override;
    
    // Print type to string representation
    void printType(Type type, DialectAsmPrinter &printer) const override;
    
    void initialize();
};

//===----------------------------------------------------------------------===//
// PostgreSQL Types
//===----------------------------------------------------------------------===//

namespace detail {
// String type storages
struct TextTypeStorage;
struct CharTypeStorage;
struct VarCharTypeStorage;

// Numeric type storages  
struct SmallIntTypeStorage;
struct IntegerTypeStorage;
struct BigIntTypeStorage;
struct NumericTypeStorage;
struct RealTypeStorage;
struct DoubleTypeStorage;
struct MoneyTypeStorage;

// Date/time type storages
struct DateTypeStorage;
struct TimeTypeStorage;
struct TimeTzTypeStorage;
struct TimestampTypeStorage;
struct TimestampTzTypeStorage;
struct IntervalTypeStorage;

// Boolean and binary type storages
struct BooleanTypeStorage;
struct ByteaTypeStorage;

// Network type storages
struct UuidTypeStorage;
struct InetTypeStorage;
struct CidrTypeStorage;
struct MacAddrTypeStorage;

// Bit type storages
struct BitTypeStorage;
struct VarBitTypeStorage;

// System type storages
struct PgTupleTypeStorage;
struct TableHandleTypeStorage;
struct TupleHandleTypeStorage;
}

/// PostgreSQL text type - variable length string
class TextType : public Type::TypeBase<TextType, Type, detail::TextTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.text";
    static TextType get(MLIRContext *context);
    static StringRef getMnemonic() { return "text"; }
};

/// PostgreSQL numeric type - arbitrary precision decimal
class NumericType : public Type::TypeBase<NumericType, Type, detail::NumericTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.numeric";
    static NumericType get(MLIRContext *context, unsigned precision = 0, unsigned scale = 0);
    static StringRef getMnemonic() { return "numeric"; }
    
    unsigned getPrecision() const;
    unsigned getScale() const;
};

/// PostgreSQL date type
class DateType : public Type::TypeBase<DateType, Type, detail::DateTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.date";
    static DateType get(MLIRContext *context);
    static StringRef getMnemonic() { return "date"; }
};

//===----------------------------------------------------------------------===//
// Integer Types
//===----------------------------------------------------------------------===//

class PgSmallIntType : public Type::TypeBase<PgSmallIntType, Type, detail::SmallIntTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.smallint";
    static PgSmallIntType get(MLIRContext *context);
    static StringRef getMnemonic() { return "smallint"; }
};

class PgIntegerType : public Type::TypeBase<PgIntegerType, Type, detail::IntegerTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.integer";
    static PgIntegerType get(MLIRContext *context);
    static StringRef getMnemonic() { return "integer"; }
};

class PgBigIntType : public Type::TypeBase<PgBigIntType, Type, detail::BigIntTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.bigint";
    static PgBigIntType get(MLIRContext *context);
    static StringRef getMnemonic() { return "bigint"; }
};

//===----------------------------------------------------------------------===//
// Decimal/Float Types
//===----------------------------------------------------------------------===//

class RealType : public Type::TypeBase<RealType, Type, detail::RealTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.real";
    static RealType get(MLIRContext *context);
    static StringRef getMnemonic() { return "real"; }
};

class DoubleType : public Type::TypeBase<DoubleType, Type, detail::DoubleTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.double";
    static DoubleType get(MLIRContext *context);
    static StringRef getMnemonic() { return "double"; }
};

class MoneyType : public Type::TypeBase<MoneyType, Type, detail::MoneyTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.money";
    static MoneyType get(MLIRContext *context);
    static StringRef getMnemonic() { return "money"; }
};

//===----------------------------------------------------------------------===//
// String Types
//===----------------------------------------------------------------------===//

class CharType : public Type::TypeBase<CharType, Type, detail::CharTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.char";
    static CharType get(MLIRContext *context, unsigned length);
    static StringRef getMnemonic() { return "char"; }
    
    unsigned getLength() const;
};

class VarCharType : public Type::TypeBase<VarCharType, Type, detail::VarCharTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.varchar";
    static VarCharType get(MLIRContext *context, unsigned maxLength);
    static StringRef getMnemonic() { return "varchar"; }
    
    unsigned getMaxLength() const;
};

//===----------------------------------------------------------------------===//
// Boolean and Binary Types
//===----------------------------------------------------------------------===//

class BooleanType : public Type::TypeBase<BooleanType, Type, detail::BooleanTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.boolean";
    static BooleanType get(MLIRContext *context);
    static StringRef getMnemonic() { return "boolean"; }
};

class ByteaType : public Type::TypeBase<ByteaType, Type, detail::ByteaTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.bytea";
    static ByteaType get(MLIRContext *context);
    static StringRef getMnemonic() { return "bytea"; }
};

//===----------------------------------------------------------------------===//
// Date/Time Types
//===----------------------------------------------------------------------===//

class TimeType : public Type::TypeBase<TimeType, Type, detail::TimeTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.time";
    static TimeType get(MLIRContext *context);
    static StringRef getMnemonic() { return "time"; }
};

class TimeTzType : public Type::TypeBase<TimeTzType, Type, detail::TimeTzTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.timetz";
    static TimeTzType get(MLIRContext *context);
    static StringRef getMnemonic() { return "timetz"; }
};

class TimestampType : public Type::TypeBase<TimestampType, Type, detail::TimestampTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.timestamp";
    static TimestampType get(MLIRContext *context);
    static StringRef getMnemonic() { return "timestamp"; }
};

class TimestampTzType : public Type::TypeBase<TimestampTzType, Type, detail::TimestampTzTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.timestamptz";
    static TimestampTzType get(MLIRContext *context);
    static StringRef getMnemonic() { return "timestamptz"; }
};

class IntervalType : public Type::TypeBase<IntervalType, Type, detail::IntervalTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.interval";
    static IntervalType get(MLIRContext *context);
    static StringRef getMnemonic() { return "interval"; }
};

//===----------------------------------------------------------------------===//
// Network Types
//===----------------------------------------------------------------------===//

class UuidType : public Type::TypeBase<UuidType, Type, detail::UuidTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.uuid";
    static UuidType get(MLIRContext *context);
    static StringRef getMnemonic() { return "uuid"; }
};

class InetType : public Type::TypeBase<InetType, Type, detail::InetTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.inet";
    static InetType get(MLIRContext *context);
    static StringRef getMnemonic() { return "inet"; }
};

class CidrType : public Type::TypeBase<CidrType, Type, detail::CidrTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.cidr";
    static CidrType get(MLIRContext *context);
    static StringRef getMnemonic() { return "cidr"; }
};

class MacAddrType : public Type::TypeBase<MacAddrType, Type, detail::MacAddrTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.macaddr";
    static MacAddrType get(MLIRContext *context);
    static StringRef getMnemonic() { return "macaddr"; }
};

//===----------------------------------------------------------------------===//
// Bit Types
//===----------------------------------------------------------------------===//

class BitType : public Type::TypeBase<BitType, Type, detail::BitTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.bit";
    static BitType get(MLIRContext *context, unsigned length);
    static StringRef getMnemonic() { return "bit"; }
    
    unsigned getLength() const;
};

class VarBitType : public Type::TypeBase<VarBitType, Type, detail::VarBitTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.varbit";
    static VarBitType get(MLIRContext *context, unsigned maxLength);
    static StringRef getMnemonic() { return "varbit"; }
    
    unsigned getMaxLength() const;
};

/// PostgreSQL tuple type - represents a row
class PgTupleType : public Type::TypeBase<PgTupleType, Type, detail::PgTupleTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.tuple";
    static PgTupleType get(MLIRContext *context, ArrayRef<Type> fieldTypes);
    static StringRef getMnemonic() { return "tuple"; }
    
    ArrayRef<Type> getFieldTypes() const;
    unsigned getNumFields() const;
    Type getFieldType(unsigned index) const;
};

/// Handle to a PostgreSQL table scan
class TableHandleType : public Type::TypeBase<TableHandleType, Type, detail::TableHandleTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.table_handle";
    static TableHandleType get(MLIRContext *context);
    static StringRef getMnemonic() { return "table_handle"; }
};

/// Handle to a PostgreSQL tuple in memory
class TupleHandleType : public Type::TypeBase<TupleHandleType, Type, detail::TupleHandleTypeStorage> {
public:
    using Base::Base;
    
    static constexpr StringRef name = "pg.tuple_handle";
    static TupleHandleType get(MLIRContext *context);
    static StringRef getMnemonic() { return "tuple_handle"; }
};

//===----------------------------------------------------------------------===//
// PostgreSQL Operations
//===----------------------------------------------------------------------===//

// Operations will be defined directly inheriting from Op for simplicity

/// Operation for scanning a PostgreSQL table
class ScanTableOp : public Op<ScanTableOp> {
public:
    using Op::Op;
    
    static constexpr StringRef getOperationName() { return "pg.scan_table"; }
    static StringRef getDialectNamespace() { return "pg"; }
    
    static void build(OpBuilder &builder, OperationState &result, 
                     StringRef tableName);
    
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &printer);
    LogicalResult verify();
    
    // Required for MLIR operation framework
    static bool classof(Operation *op) {
        return op->getName().getStringRef() == getOperationName();
    }
    
    static ArrayRef<StringRef> getAttributeNames() { 
        static constexpr StringRef attrNames[] = {"table_name"};
        return llvm::ArrayRef(attrNames);
    }
    
    // Get the table name attribute
    StringRef getTableName() { 
        return (*this)->getAttrOfType<StringAttr>("table_name").getValue();
    }
    
    // Get the result as a table handle
    Value getResult() {
        return (*this)->getResult(0);
    }
};

/// Operation for reading the next tuple from a table handle
class ReadTupleOp : public Op<ReadTupleOp> {
public:
    using Op::Op;
    
    static constexpr StringRef getOperationName() { return "pg.read_tuple"; }
    static StringRef getDialectNamespace() { return "pg"; }
    
    static void build(OpBuilder &builder, OperationState &result, 
                     Value tableHandle, ArrayRef<Type> fieldTypes);
    
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &printer);
    LogicalResult verify();
    
    // Required for MLIR operation framework
    static bool classof(Operation *op) {
        return op->getName().getStringRef() == getOperationName();
    }
    
    static ArrayRef<StringRef> getAttributeNames() { 
        return llvm::ArrayRef<StringRef>();
    }
    
    // Get the table handle operand
    Value getTableHandle() {
        return (*this)->getOperand(0);
    }
    
    // Get the result tuple
    Value getResult() {
        return (*this)->getResult(0);
    }
};

/// Operation for getting an integer field from a tuple
class GetIntFieldOp : public Op<GetIntFieldOp> {
public:
    using Op::Op;
    
    static constexpr StringRef getOperationName() { return "pg.get_int_field"; }
    static StringRef getDialectNamespace() { return "pg"; }
    
    static void build(OpBuilder &builder, OperationState &result, 
                     Value tuple, unsigned fieldIndex);
    
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &printer);
    LogicalResult verify();
    
    // Required for MLIR operation framework
    static bool classof(Operation *op) {
        return op->getName().getStringRef() == getOperationName();
    }
    
    static ArrayRef<StringRef> getAttributeNames() { 
        static constexpr StringRef attrNames[] = {"field_index"};
        return llvm::ArrayRef(attrNames);
    }
    
    // Get the tuple operand
    Value getTuple() {
        return (*this)->getOperand(0);
    }
    
    // Get the field index
    unsigned getFieldIndex() {
        return (*this)->getAttrOfType<IntegerAttr>("field_index").getValue().getZExtValue();
    }
    
    // Get the result value (int32)
    Value getResult() {
        return (*this)->getResult(0);
    }
    
    // Get the null flag result (i1)
    Value getNullFlag() {
        return (*this)->getResult(1);
    }
};

/// Operation for getting a text field from a tuple  
class GetTextFieldOp : public Op<GetTextFieldOp> {
public:
    using Op::Op;
    
    static constexpr StringRef getOperationName() { return "pg.get_text_field"; }
    static StringRef getDialectNamespace() { return "pg"; }
    
    static void build(OpBuilder &builder, OperationState &result, 
                     Value tuple, unsigned fieldIndex);
    
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &printer);
    LogicalResult verify();
    
    // Required for MLIR operation framework
    static bool classof(Operation *op) {
        return op->getName().getStringRef() == getOperationName();
    }
    
    static ArrayRef<StringRef> getAttributeNames() { 
        static constexpr StringRef attrNames[] = {"field_index"};
        return llvm::ArrayRef(attrNames);
    }
    
    // Get the tuple operand
    Value getTuple() {
        return (*this)->getOperand(0);
    }
    
    // Get the field index
    unsigned getFieldIndex() {
        return (*this)->getAttrOfType<IntegerAttr>("field_index").getValue().getZExtValue();
    }
    
    // Get the result value (text as i64 pointer)
    Value getResult() {
        return (*this)->getResult(0);
    }
    
    // Get the null flag result (i1)
    Value getNullFlag() {
        return (*this)->getResult(1);
    }
};

} // namespace pg
} // namespace mlir

// TODO: Include auto-generated headers when TableGen is fixed
// #include "dialects/pg/PgOps.h.inc"
// #define GET_TYPEDEF_CLASSES
// #include "dialects/pg/PgTypes.h.inc"