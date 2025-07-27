//===----------------------------------------------------------------------===//
// PostgreSQL Dialect Operations
//===----------------------------------------------------------------------===//

#ifndef PG_OPS
#define PG_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "dialects/pg/PgBase.td"
include "dialects/pg/PgTypes.td"
include "dialects/pg/PgOperators.td"

//===----------------------------------------------------------------------===//
// Base Operation Class (defined in PgBase.td)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Table Operations
//===----------------------------------------------------------------------===//

def Pg_ScanTableOp : Pg_Op<"scan_table", [Pure]> {
    let summary = "Scan a PostgreSQL table";

    let arguments = (ins StrAttr:$table_name);
    let results = (outs Pg_TableHandleType:$handle);
    
    let assemblyFormat = "$table_name attr-dict `:` type($handle)";
}

// RelAlg-style base table operation
def Pg_BaseTableOp : Pg_Op<"basetable", [Pure]> {
    let summary = "Base table operation - RelAlg-style table access";

    let arguments = (ins StrAttr:$table_identifier);
    let results = (outs Pg_TupleHandleType:$stream);
    
    let assemblyFormat = "`{` `table_identifier` `=` $table_identifier `}` attr-dict `:` type($stream)";
}

// RelAlg-style materialization operation
def Pg_MaterializeTupleOp : Pg_Op<"materialize_tuple", []> {
    let summary = "Materialize tuple to query results - RelAlg-style";

    let arguments = (ins Pg_TupleHandleType:$tuple);
    let results = (outs);
    
    let assemblyFormat = "$tuple attr-dict `:` type($tuple)";
}

def Pg_ReadTupleOp : Pg_Op<"read_tuple", [Pure]> {
    let summary = "Read next tuple from table scan";

    let arguments = (ins Pg_TableHandleType:$table_handle);
    let results = (outs Pg_TupleHandleType:$tuple);
    
    let assemblyFormat = "$table_handle attr-dict `:` type($table_handle) `->` type($tuple)";
}

//===----------------------------------------------------------------------===//
// Field Access Operations
//===----------------------------------------------------------------------===//

def Pg_GetIntFieldOp : Pg_Op<"get_int_field", [Pure]> {
    let summary = "Extract integer field from tuple";

    let arguments = (ins Pg_TupleHandleType:$tuple, I32Attr:$field_index);
    let results = (outs I32:$value, I1:$is_null);
    
    let assemblyFormat = "$tuple `,` $field_index attr-dict `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`";
}

def Pg_GetTextFieldOp : Pg_Op<"get_text_field", [Pure]> {
    let summary = "Extract text field from tuple";

    let arguments = (ins Pg_TupleHandleType:$tuple, I32Attr:$field_index);
    let results = (outs I64:$value, I1:$is_null);
    
    let assemblyFormat = "$tuple `,` $field_index attr-dict `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`";
}

def Pg_GetNumericFieldOp : Pg_Op<"get_numeric_field", [Pure]> {
    let summary = "Extract numeric field from tuple";

    let arguments = (ins Pg_TupleType:$tuple, I32Attr:$field_index);
    let results = (outs Pg_NumericType:$value, I1:$is_null);
    
    let assemblyFormat = "$tuple `[` $field_index `]` attr-dict `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`";
}

//===----------------------------------------------------------------------===//
// Predicate Operations
//===----------------------------------------------------------------------===//

def Pg_WhereOp : Pg_Op<"where", [Pure]> {
    let summary = "Apply WHERE clause predicate";

    let arguments = (ins Pg_TupleType:$tuple, I1:$condition, DefaultValuedAttr<StrAttr, "\"reject\"">:$null_handling);
    let results = (outs Pg_TupleType:$result);
    
    let assemblyFormat = "$tuple `,` $condition attr-dict `:` type($tuple) `,` type($condition) `->` type($result)";
}

def Pg_IsNullOp : Pg_Op<"is_null", [Pure]> {
    let summary = "Check if a value is NULL";

    let arguments = (ins AnyType:$value, DefaultValuedAttr<BoolAttr, "false">:$invert);
    let results = (outs I1:$result);
    
    let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def Pg_CmpIntOp : Pg_Op<"cmp_int", [Pure]> {
    let summary = "Compare two integer values";

    let arguments = (ins I32:$lhs, I32:$rhs, StrAttr:$predicate);
    let results = (outs I1:$result);
    
    let assemblyFormat = "$lhs `,` $rhs $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Pg_CmpTextOp : Pg_Op<"cmp_text", [Pure]> {
    let summary = "Compare two text values";

    let arguments = (ins Pg_TextType:$lhs, Pg_TextType:$rhs, StrAttr:$predicate);
    let results = (outs I1:$result);
    
    let assemblyFormat = "$lhs `,` $rhs $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Aggregation Operations
//===----------------------------------------------------------------------===//

def Pg_AggregateSumOp : Pg_Op<"aggregate_sum", [Pure]> {
    let summary = "Sum aggregation operation";

    let arguments = (ins Pg_TableHandleType:$input, DefaultValuedAttr<BoolAttr, "false">:$distinct);
    let results = (outs I64:$result);
    
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Pg_AggregateCountOp : Pg_Op<"aggregate_count", [Pure]> {
    let summary = "Count aggregation operation";

    let arguments = (ins Pg_TableHandleType:$input, DefaultValuedAttr<BoolAttr, "false">:$distinct);
    let results = (outs I64:$result);
    
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Output Operations
//===----------------------------------------------------------------------===//

def Pg_YieldTupleOp : Pg_Op<"yield_tuple", [Pure]> {
    let summary = "Yield tuple to query result";

    let arguments = (ins Pg_TupleType:$tuple, DefaultValuedAttr<StrAttr, "\"none\"">:$ordering);
    let results = (outs);
    
    let assemblyFormat = "$tuple attr-dict `:` type($tuple)";
}

#endif // PG_OPS