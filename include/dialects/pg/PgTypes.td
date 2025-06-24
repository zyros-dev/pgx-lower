//===----------------------------------------------------------------------===//
// PostgreSQL Dialect Types
//===----------------------------------------------------------------------===//

#ifndef PG_TYPES
#define PG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "dialects/pg/PgBase.td"

//===----------------------------------------------------------------------===//
// PostgreSQL Type Definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Basic PostgreSQL Types
//===----------------------------------------------------------------------===//

def Pg_TextType : Pg_Type<"Text", "text"> {
    let summary = "PostgreSQL TEXT type";
    let description = [{
        Represents a PostgreSQL TEXT value - variable length string.
        Maps to PostgreSQL's text/varchar types.
    }];
    let extraClassDeclaration = [{
        static TextType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_NumericType : Pg_Type<"Numeric", "numeric"> {
    let summary = "PostgreSQL NUMERIC type";
    let description = [{
        Represents a PostgreSQL NUMERIC value - arbitrary precision decimal.
        Maps to PostgreSQL's numeric/decimal types.
    }];
}

def Pg_DateType : Pg_Type<"Date", "date"> {
    let summary = "PostgreSQL DATE type";
    let description = [{
        Represents a PostgreSQL DATE value.
        Maps to PostgreSQL's date type.
    }];
}

def Pg_TimestampType : Pg_Type<"Timestamp", "timestamp"> {
    let summary = "PostgreSQL TIMESTAMP type";
    let description = [{
        Represents a PostgreSQL TIMESTAMP value.
        Maps to PostgreSQL's timestamp types.
    }];
}

def Pg_BoolType : Pg_Type<"Bool", "bool"> {
    let summary = "PostgreSQL BOOLEAN type";
    let description = [{
        Represents a PostgreSQL BOOLEAN value.
        Maps to PostgreSQL's boolean type.
    }];
}

//===----------------------------------------------------------------------===//
// Composite Types
//===----------------------------------------------------------------------===//

def Pg_TupleType : Pg_Type<"Tuple", "tuple"> {
    let summary = "PostgreSQL tuple type";
    let description = [{
        Represents a PostgreSQL tuple (row) containing multiple fields
        of potentially different types.
    }];
}

def Pg_TableHandleType : Pg_Type<"TableHandle", "table_handle"> {
    let summary = "Handle to a PostgreSQL table scan";
    let description = [{
        Represents an active scan of a PostgreSQL table.
        This is an opaque handle used for tuple iteration.
    }];
    let extraClassDeclaration = [{
        static TableHandleType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TupleHandleType : Pg_Type<"TupleHandle", "tuple_handle"> {
    let summary = "Handle to a PostgreSQL tuple";
    let description = [{
        Represents a PostgreSQL tuple (row) with access to its
        fields and metadata. Supports type-safe field access operations.
    }];
    let extraClassDeclaration = [{
        static TupleHandleType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def Pg_AnyType : AnyTypeOf<[
    Pg_TextType,
    Pg_NumericType,
    Pg_DateType,
    Pg_TimestampType,
    Pg_BoolType,
    I32,
    I64
]>;

def Pg_ComparableType : AnyTypeOf<[
    Pg_TextType,
    Pg_NumericType,
    Pg_DateType,
    Pg_TimestampType,
    I32,
    I64
]>;

#endif // PG_TYPES