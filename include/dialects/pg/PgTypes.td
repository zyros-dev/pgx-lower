//===----------------------------------------------------------------------===//
// PostgreSQL Dialect Types
//===----------------------------------------------------------------------===//

#ifndef PG_TYPES
#define PG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "dialects/pg/PgDialect.td"

//===----------------------------------------------------------------------===//
// PostgreSQL Type Definitions
//===----------------------------------------------------------------------===//

class Pg_Type<string name, string typeMnemonic> : TypeDef<Pg_Dialect, name> {
    let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Basic PostgreSQL Types
//===----------------------------------------------------------------------===//

def Pg_TextType : Pg_Type<"Text", "text"> {
    let summary = "PostgreSQL TEXT type";
    let description = [{
        Represents a PostgreSQL TEXT value - variable length string.
        Maps to PostgreSQL's text/varchar types.
    }];
}

def Pg_NumericType : Pg_Type<"Numeric", "numeric"> {
    let summary = "PostgreSQL NUMERIC type";
    let description = [{
        Represents a PostgreSQL NUMERIC value - arbitrary precision decimal.
        Maps to PostgreSQL's numeric/decimal types.
    }];
    
    let parameters = (ins 
        "unsigned":$precision,
        "unsigned":$scale
    );
    
    let assemblyFormat = "`<` $precision `,` $scale `>`";
    
    let builders = [
        TypeBuilder<(ins), [{
            return $_get($_ctxt, 0, 0);
        }]>,
        TypeBuilder<(ins "unsigned":$precision, "unsigned":$scale), [{
            return $_get($_ctxt, precision, scale);
        }]>
    ];
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
    
    let parameters = (ins ArrayRefParameter<"Type", "field types">:$fieldTypes);
    
    let assemblyFormat = "`<` $fieldTypes `>`";
    
    let builders = [
        TypeBuilder<(ins "ArrayRef<Type>":$fieldTypes), [{
            return $_get($_ctxt, fieldTypes);
        }]>
    ];
    
    let extraClassDeclaration = [{
        unsigned getNumFields() const { return getFieldTypes().size(); }
        Type getFieldType(unsigned index) const { 
            return getFieldTypes()[index]; 
        }
    }];
}

def Pg_TableHandleType : Pg_Type<"TableHandle", "table_handle"> {
    let summary = "Handle to a PostgreSQL table scan";
    let description = [{
        Represents an active scan of a PostgreSQL table.
        This is an opaque handle used for tuple iteration.
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