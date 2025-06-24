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
    let assemblyFormat = "";
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
    let parameters = (ins "unsigned":$precision, "unsigned":$scale);
    let assemblyFormat = "`<` $precision `,` $scale `>`";
}

def Pg_DateType : Pg_Type<"Date", "date"> {
    let summary = "PostgreSQL DATE type";
    let description = [{
        Represents a PostgreSQL DATE value.
        Maps to PostgreSQL's date type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static DateType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TimestampType : Pg_Type<"Timestamp", "timestamp"> {
    let summary = "PostgreSQL TIMESTAMP type";
    let description = [{
        Represents a PostgreSQL TIMESTAMP value.
        Maps to PostgreSQL's timestamp types.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimestampType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_BooleanType : Pg_Type<"Boolean", "boolean"> {
    let summary = "PostgreSQL BOOLEAN type";
    let description = [{
        Represents a PostgreSQL BOOLEAN value.
        Maps to PostgreSQL's boolean type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static BooleanType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// String Types
//===----------------------------------------------------------------------===//

def Pg_CharType : Pg_Type<"Char", "char"> {
    let summary = "PostgreSQL CHAR type";
    let description = [{
        Represents a PostgreSQL CHAR value - fixed length string.
        Maps to PostgreSQL's char(n) type.
    }];
    let parameters = (ins "unsigned":$length);
    let assemblyFormat = "`<` $length `>`";
}

def Pg_VarCharType : Pg_Type<"VarChar", "varchar"> {
    let summary = "PostgreSQL VARCHAR type";
    let description = [{
        Represents a PostgreSQL VARCHAR value - variable length string with max length.
        Maps to PostgreSQL's varchar(n) type.
    }];
    let parameters = (ins "unsigned":$maxLength);
    let assemblyFormat = "`<` $maxLength `>`";
}

//===----------------------------------------------------------------------===//
// Integer Types
//===----------------------------------------------------------------------===//

def Pg_SmallIntType : Pg_Type<"PgSmallInt", "smallint"> {
    let summary = "PostgreSQL SMALLINT type";
    let description = [{
        Represents a PostgreSQL SMALLINT value - 16-bit signed integer.
        Maps to PostgreSQL's smallint type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static PgSmallIntType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_IntegerType : Pg_Type<"PgInteger", "integer"> {
    let summary = "PostgreSQL INTEGER type";
    let description = [{
        Represents a PostgreSQL INTEGER value - 32-bit signed integer.
        Maps to PostgreSQL's integer type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static PgIntegerType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_BigIntType : Pg_Type<"PgBigInt", "bigint"> {
    let summary = "PostgreSQL BIGINT type";
    let description = [{
        Represents a PostgreSQL BIGINT value - 64-bit signed integer.
        Maps to PostgreSQL's bigint type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static PgBigIntType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// Decimal/Float Types
//===----------------------------------------------------------------------===//

def Pg_RealType : Pg_Type<"Real", "real"> {
    let summary = "PostgreSQL REAL type";
    let description = [{
        Represents a PostgreSQL REAL value - single precision floating point.
        Maps to PostgreSQL's real type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static RealType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_DoubleType : Pg_Type<"Double", "double"> {
    let summary = "PostgreSQL DOUBLE PRECISION type";
    let description = [{
        Represents a PostgreSQL DOUBLE PRECISION value - double precision floating point.
        Maps to PostgreSQL's double precision type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static DoubleType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_MoneyType : Pg_Type<"Money", "money"> {
    let summary = "PostgreSQL MONEY type";
    let description = [{
        Represents a PostgreSQL MONEY value - currency amount.
        Maps to PostgreSQL's money type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static MoneyType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// Binary Types
//===----------------------------------------------------------------------===//

def Pg_ByteaType : Pg_Type<"Bytea", "bytea"> {
    let summary = "PostgreSQL BYTEA type";
    let description = [{
        Represents a PostgreSQL BYTEA value - binary data.
        Maps to PostgreSQL's bytea type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static ByteaType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// Date/Time Types
//===----------------------------------------------------------------------===//

def Pg_TimeType : Pg_Type<"Time", "time"> {
    let summary = "PostgreSQL TIME type";
    let description = [{
        Represents a PostgreSQL TIME value - time of day without timezone.
        Maps to PostgreSQL's time type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimeType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TimeTzType : Pg_Type<"TimeTz", "timetz"> {
    let summary = "PostgreSQL TIME WITH TIME ZONE type";
    let description = [{
        Represents a PostgreSQL TIME WITH TIME ZONE value.
        Maps to PostgreSQL's time with time zone type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimeTzType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TimestampTzType : Pg_Type<"TimestampTz", "timestamptz"> {
    let summary = "PostgreSQL TIMESTAMP WITH TIME ZONE type";
    let description = [{
        Represents a PostgreSQL TIMESTAMP WITH TIME ZONE value.
        Maps to PostgreSQL's timestamp with time zone type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimestampTzType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_IntervalType : Pg_Type<"Interval", "interval"> {
    let summary = "PostgreSQL INTERVAL type";
    let description = [{
        Represents a PostgreSQL INTERVAL value - time span.
        Maps to PostgreSQL's interval type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static IntervalType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// Network Types
//===----------------------------------------------------------------------===//

def Pg_UuidType : Pg_Type<"Uuid", "uuid"> {
    let summary = "PostgreSQL UUID type";
    let description = [{
        Represents a PostgreSQL UUID value - universally unique identifier.
        Maps to PostgreSQL's uuid type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static UuidType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_InetType : Pg_Type<"Inet", "inet"> {
    let summary = "PostgreSQL INET type";
    let description = [{
        Represents a PostgreSQL INET value - IP address.
        Maps to PostgreSQL's inet type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static InetType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_CidrType : Pg_Type<"Cidr", "cidr"> {
    let summary = "PostgreSQL CIDR type";
    let description = [{
        Represents a PostgreSQL CIDR value - network address.
        Maps to PostgreSQL's cidr type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static CidrType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_MacAddrType : Pg_Type<"MacAddr", "macaddr"> {
    let summary = "PostgreSQL MACADDR type";
    let description = [{
        Represents a PostgreSQL MACADDR value - MAC address.
        Maps to PostgreSQL's macaddr type.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static MacAddrType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

//===----------------------------------------------------------------------===//
// Bit Types
//===----------------------------------------------------------------------===//

def Pg_BitType : Pg_Type<"Bit", "bit"> {
    let summary = "PostgreSQL BIT type";
    let description = [{
        Represents a PostgreSQL BIT value - fixed length bit string.
        Maps to PostgreSQL's bit(n) type.
    }];
    let parameters = (ins "unsigned":$length);
    let assemblyFormat = "`<` $length `>`";
}

def Pg_VarBitType : Pg_Type<"VarBit", "varbit"> {
    let summary = "PostgreSQL VARBIT type";
    let description = [{
        Represents a PostgreSQL VARBIT value - variable length bit string.
        Maps to PostgreSQL's varbit(n) type.
    }];
    let parameters = (ins "unsigned":$maxLength);
    let assemblyFormat = "`<` $maxLength `>`";
}

//===----------------------------------------------------------------------===//
// Composite Types
//===----------------------------------------------------------------------===//

def Pg_TupleType : Pg_Type<"PgTuple", "tuple"> {
    let summary = "PostgreSQL tuple type";
    let description = [{
        Represents a PostgreSQL tuple (row) containing multiple fields
        of potentially different types.
    }];
    let parameters = (ins ArrayRefParameter<"Type", "field types">:$fieldTypes);
    let assemblyFormat = "`<` $fieldTypes `>`";
}

def Pg_TableHandleType : Pg_Type<"TableHandle", "table_handle"> {
    let summary = "Handle to a PostgreSQL table scan";
    let description = [{
        Represents an active scan of a PostgreSQL table.
        This is an opaque handle used for tuple iteration.
    }];
    let assemblyFormat = "";
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
    let assemblyFormat = "";
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
    Pg_CharType,
    Pg_VarCharType,
    Pg_SmallIntType,
    Pg_IntegerType,
    Pg_BigIntType,
    Pg_NumericType,
    Pg_RealType,
    Pg_DoubleType,
    Pg_MoneyType,
    Pg_BooleanType,
    Pg_ByteaType,
    Pg_DateType,
    Pg_TimeType,
    Pg_TimeTzType,
    Pg_TimestampType,
    Pg_TimestampTzType,
    Pg_IntervalType,
    Pg_UuidType,
    Pg_InetType,
    Pg_CidrType,
    Pg_MacAddrType,
    Pg_BitType,
    Pg_VarBitType,
    Pg_TupleType,
    Pg_TableHandleType,
    Pg_TupleHandleType,
    I32,
    I64
]>;

def Pg_ComparableType : AnyTypeOf<[
    Pg_TextType,
    Pg_CharType,
    Pg_VarCharType,
    Pg_SmallIntType,
    Pg_IntegerType,
    Pg_BigIntType,
    Pg_NumericType,
    Pg_RealType,
    Pg_DoubleType,
    Pg_MoneyType,
    Pg_DateType,
    Pg_TimeType,
    Pg_TimeTzType,
    Pg_TimestampType,
    Pg_TimestampTzType,
    I32,
    I64
]>;

#endif // PG_TYPES