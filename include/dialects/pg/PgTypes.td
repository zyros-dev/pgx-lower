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
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TextType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_NumericType : Pg_Type<"Numeric", "numeric"> {
    let summary = "PostgreSQL NUMERIC type";
    let parameters = (ins "unsigned":$precision, "unsigned":$scale);
    let assemblyFormat = "`<` $precision `,` $scale `>`";
}

def Pg_DateType : Pg_Type<"Date", "date"> {
    let summary = "PostgreSQL DATE type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static DateType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TimestampType : Pg_Type<"Timestamp", "timestamp"> {
    let summary = "PostgreSQL TIMESTAMP type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimestampType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_BooleanType : Pg_Type<"Boolean", "boolean"> {
    let summary = "PostgreSQL BOOLEAN type";
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
    let parameters = (ins "unsigned":$length);
    let assemblyFormat = "`<` $length `>`";
}

def Pg_VarCharType : Pg_Type<"VarChar", "varchar"> {
    let summary = "PostgreSQL VARCHAR type";
    let parameters = (ins "unsigned":$maxLength);
    let assemblyFormat = "`<` $maxLength `>`";
}

//===----------------------------------------------------------------------===//
// Integer Types
//===----------------------------------------------------------------------===//

def Pg_SmallIntType : Pg_Type<"PgSmallInt", "smallint"> {
    let summary = "PostgreSQL SMALLINT type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static PgSmallIntType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_IntegerType : Pg_Type<"PgInteger", "integer"> {
    let summary = "PostgreSQL INTEGER type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static PgIntegerType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_BigIntType : Pg_Type<"PgBigInt", "bigint"> {
    let summary = "PostgreSQL BIGINT type";
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
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static RealType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_DoubleType : Pg_Type<"Double", "double"> {
    let summary = "PostgreSQL DOUBLE PRECISION type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static DoubleType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_MoneyType : Pg_Type<"Money", "money"> {
    let summary = "PostgreSQL MONEY type";
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
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimeType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TimeTzType : Pg_Type<"TimeTz", "timetz"> {
    let summary = "PostgreSQL TIME WITH TIME ZONE type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimeTzType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TimestampTzType : Pg_Type<"TimestampTz", "timestamptz"> {
    let summary = "PostgreSQL TIMESTAMP WITH TIME ZONE type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TimestampTzType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_IntervalType : Pg_Type<"Interval", "interval"> {
    let summary = "PostgreSQL INTERVAL type";
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
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static UuidType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_InetType : Pg_Type<"Inet", "inet"> {
    let summary = "PostgreSQL INET type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static InetType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_CidrType : Pg_Type<"Cidr", "cidr"> {
    let summary = "PostgreSQL CIDR type";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static CidrType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_MacAddrType : Pg_Type<"MacAddr", "macaddr"> {
    let summary = "PostgreSQL MACADDR type";
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
    let parameters = (ins "unsigned":$length);
    let assemblyFormat = "`<` $length `>`";
}

def Pg_VarBitType : Pg_Type<"VarBit", "varbit"> {
    let summary = "PostgreSQL VARBIT type";
    let parameters = (ins "unsigned":$maxLength);
    let assemblyFormat = "`<` $maxLength `>`";
}

//===----------------------------------------------------------------------===//
// Composite Types
//===----------------------------------------------------------------------===//

def Pg_TupleType : Pg_Type<"PgTuple", "tuple"> {
    let summary = "PostgreSQL tuple type";
    let parameters = (ins ArrayRefParameter<"Type", "field types">:$fieldTypes);
    let assemblyFormat = "`<` $fieldTypes `>`";
}

def Pg_TableHandleType : Pg_Type<"TableHandle", "table_handle"> {
    let summary = "Handle to a PostgreSQL table scan";
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        static TableHandleType get(::mlir::MLIRContext *context) {
            return Base::get(context);
        }
    }];
}

def Pg_TupleHandleType : Pg_Type<"TupleHandle", "tuple_handle"> {
    let summary = "Handle to a PostgreSQL tuple";
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