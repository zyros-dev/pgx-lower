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
struct TextTypeStorage;
struct NumericTypeStorage;
struct DateTypeStorage;
struct PgTupleTypeStorage;
struct TableHandleTypeStorage;
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

//===----------------------------------------------------------------------===//
// PostgreSQL Operations
//===----------------------------------------------------------------------===//

/// Base class for all PostgreSQL operations
class PgOp : public Op<PgOp> {
public:
    using Op::Op;
    
    static ArrayRef<StringRef> getAttributeNames() { return {}; }
    
    static StringRef getDialectName() { return "pg"; }
};

} // namespace pg
} // namespace mlir

// TODO: Include auto-generated headers when TableGen is set up
// #include "dialects/pg/PgOps.h.inc"
// #define GET_TYPEDEF_CLASSES
// #include "dialects/pg/PgTypes.h.inc"