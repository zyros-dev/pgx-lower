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
    // TODO: Add operations and types when TableGen is set up
    // addOperations<...>();
    addTypes<TextType, NumericType, DateType, PgTupleType, TableHandleType>();
}

PgDialect::PgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PgDialect>()) {
    initialize();
}

Type PgDialect::parseType(DialectAsmParser &parser) const {
    // TODO: Implement manual type parser until TableGen is set up
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return Type();
    
    if (keyword == "text")
        return TextType::get(getContext());
    if (keyword == "date")
        return DateType::get(getContext());
    if (keyword == "table_handle")
        return TableHandleType::get(getContext());
    
    return Type();
}

void PgDialect::printType(Type type, DialectAsmPrinter &printer) const {
    // TODO: Implement manual type printer until TableGen is set up
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

// TODO: Include auto-generated definitions when TableGen is set up
// #define GET_OP_CLASSES
// #include "dialects/pg/PgOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "dialects/pg/PgTypes.cpp.inc"