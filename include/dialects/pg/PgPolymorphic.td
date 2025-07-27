//===----------------------------------------------------------------------===//
// PostgreSQL Polymorphic Operations
//===----------------------------------------------------------------------===//

#ifndef PG_POLYMORPHIC_OPS
#define PG_POLYMORPHIC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "dialects/pg/PgBase.td"
include "dialects/pg/PgTypes.td"

//===----------------------------------------------------------------------===//
// Polymorphic Field Access
//===----------------------------------------------------------------------===//

def Pg_GetFieldOp : Pg_Op<"get_field", [Pure, 
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Extract field from tuple (polymorphic)";
    let description = [{
        Extracts a field from a PostgreSQL tuple. The result type is inferred
        based on the field's PostgreSQL type OID at runtime.
    }];

    let arguments = (ins 
        Pg_TupleHandleType:$tuple, 
        I32Attr:$field_index,
        I32Attr:$field_type_oid  // PostgreSQL type OID
    );
    let results = (outs AnyType:$value, I1:$is_null);
    
    let assemblyFormat = [{
        $tuple `[` $field_index `]` `type` `(` $field_type_oid `)` attr-dict 
        `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`
    }];
}

//===----------------------------------------------------------------------===//
// Polymorphic Arithmetic Operations
//===----------------------------------------------------------------------===//

class Pg_BinaryOp<string mnemonic, list<Trait> traits = []> :
        Pg_Op<mnemonic, !listconcat(traits, [Pure, 
              DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    
    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` 
        `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

def Pg_AddOp : Pg_BinaryOp<"add"> {
    let summary = "Polymorphic addition";
    let description = [{
        Adds two values. Works with integers, floats, and numeric types.
        The result type is inferred from the operand types.
    }];
}

def Pg_SubOp : Pg_BinaryOp<"sub"> {
    let summary = "Polymorphic subtraction";
}

def Pg_MulOp : Pg_BinaryOp<"mul"> {
    let summary = "Polymorphic multiplication";
}

def Pg_DivOp : Pg_BinaryOp<"div"> {
    let summary = "Polymorphic division";
}

//===----------------------------------------------------------------------===//
// Polymorphic Comparison Operations
//===----------------------------------------------------------------------===//

def Pg_CmpOp : Pg_Op<"cmp", [Pure]> {
    let summary = "Polymorphic comparison";
    let description = [{
        Compares two values of the same type. The predicate determines
        the comparison operation (eq, ne, lt, le, gt, ge).
    }];

    let arguments = (ins 
        AnyType:$lhs, 
        AnyType:$rhs, 
        StrAttr:$predicate
    );
    let results = (outs I1:$result);
    
    let assemblyFormat = [{
        $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
    }];
}

//===----------------------------------------------------------------------===//
// Polymorphic Aggregate Operations
//===----------------------------------------------------------------------===//

def Pg_MinOp : Pg_BinaryOp<"min"> {
    let summary = "Polymorphic minimum";
    let description = [{
        Returns the minimum of two values. Works with any comparable type.
    }];
}

def Pg_MaxOp : Pg_BinaryOp<"max"> {
    let summary = "Polymorphic maximum";
    let description = [{
        Returns the maximum of two values. Works with any comparable type.
    }];
}

#endif // PG_POLYMORPHIC_OPS