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
    let description = [{
        Creates a table handle for scanning a PostgreSQL table.
        
        Example:
        ```mlir
        %table = pg.scan_table "employees" : !pg.table_handle
        ```
    }];
    
    let arguments = (ins StrAttr:$table_name);
    let results = (outs Pg_TableHandleType:$handle);
    
    let assemblyFormat = "$table_name attr-dict `:` type($handle)";
}

def Pg_ReadTupleOp : Pg_Op<"read_tuple", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Read next tuple from table scan";
    let description = [{
        Reads the next tuple from a table scan. Returns the tuple handle.
        
        Example:
        ```mlir
        %tuple = pg.read_tuple %table {limit = 1} : !pg.tuple_handle
        ```
    }];
    
    let arguments = (ins Pg_TableHandleType:$table_handle, DefaultValuedAttr<I32Attr, "1">:$limit);
    let results = (outs Pg_TupleHandleType:$tuple);
    
    let assemblyFormat = "$table_handle attr-dict `:` type($table_handle) `->` type($tuple)";
    
    let builders = [
        OpBuilder<(ins "::mlir::Value":$table_handle, "::llvm::ArrayRef<::mlir::Type>":$resultTypes), [{
            build(odsBuilder, odsState, resultTypes, table_handle, odsBuilder.getI32IntegerAttr(1));
        }]>
    ];
}

//===----------------------------------------------------------------------===//
// Field Access Operations
//===----------------------------------------------------------------------===//

def Pg_GetIntFieldOp : Pg_Op<"get_int_field", [Pure]> {
    let summary = "Extract integer field from tuple";
    let description = [{
        Extracts an integer field from a PostgreSQL tuple at the specified index.
        Handles NULL values by returning both the value and a null indicator.
        
        Example:
        ```mlir
        %value, %is_null = pg.get_int_field %tuple, 0 : !pg.tuple_handle -> (i32, i1)
        ```
    }];
    
    let arguments = (ins Pg_TupleHandleType:$tuple, I32Attr:$field_index);
    let results = (outs I32:$value, I1:$is_null);
    
    let assemblyFormat = "$tuple `,` $field_index attr-dict `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`";
}

def Pg_GetTextFieldOp : Pg_Op<"get_text_field", [Pure]> {
    let summary = "Extract text field from tuple";
    let description = [{
        Extracts a text field from a PostgreSQL tuple at the specified index.
        
        Example:
        ```mlir
        %text, %is_null = pg.get_text_field %tuple, 1 : !pg.tuple_handle -> (i64, i1)
        ```
    }];
    
    let arguments = (ins Pg_TupleHandleType:$tuple, I32Attr:$field_index);
    let results = (outs I64:$value, I1:$is_null);
    
    let assemblyFormat = "$tuple `,` $field_index attr-dict `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`";
}

def Pg_GetNumericFieldOp : Pg_Op<"get_numeric_field", [Pure]> {
    let summary = "Extract numeric field from tuple";
    let description = [{
        Extracts a numeric field from a PostgreSQL tuple at the specified index.
        
        Example:
        ```mlir
        %numeric, %is_null = pg.get_numeric_field %tuple[2] : !pg.tuple<i32, !pg.text, !pg.numeric> -> (!pg.numeric, i1)
        ```
    }];
    
    let arguments = (ins Pg_TupleType:$tuple, I32Attr:$field_index);
    let results = (outs Pg_NumericType:$value, I1:$is_null);
    
    let assemblyFormat = "$tuple `[` $field_index `]` attr-dict `:` type($tuple) `->` `(` type($value) `,` type($is_null) `)`";
}

//===----------------------------------------------------------------------===//
// Predicate Operations
//===----------------------------------------------------------------------===//

def Pg_WhereOp : Pg_Op<"where", [Pure]> {
    let summary = "Apply WHERE clause predicate";
    let description = [{
        Applies a boolean predicate to filter tuples. This is a high-level
        operation that will be lowered to conditional control flow.
        
        Example:
        ```mlir
        %filtered = pg.where %tuple, %condition {null_handling = "reject"} : !pg.tuple<i32, !pg.text>, i1 -> !pg.tuple<i32, !pg.text>
        ```
    }];
    
    let arguments = (ins Pg_TupleType:$tuple, I1:$condition, DefaultValuedAttr<StrAttr, "\"reject\"">:$null_handling);
    let results = (outs Pg_TupleType:$result);
    
    let assemblyFormat = "$tuple `,` $condition attr-dict `:` type($tuple) `,` type($condition) `->` type($result)";
}

def Pg_IsNullOp : Pg_Op<"is_null", [Pure]> {
    let summary = "Check if a value is NULL";
    let description = [{
        Checks if a PostgreSQL value is NULL. Works with any PostgreSQL type.
        Supports both IS NULL and IS NOT NULL checks.
        
        Example:
        ```mlir
        %is_null = pg.is_null %text_value {invert = false} : !pg.text -> i1
        %is_not_null = pg.is_null %text_value {invert = true} : !pg.text -> i1
        ```
    }];
    
    let arguments = (ins AnyType:$value, DefaultValuedAttr<BoolAttr, "false">:$invert);
    let results = (outs I1:$result);
    
    let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def Pg_CmpIntOp : Pg_Op<"cmp_int", [Pure]> {
    let summary = "Compare two integer values";
    let description = [{
        Compares two integer values with PostgreSQL semantics, handling NULL values.
        
        Example:
        ```mlir
        %result = pg.cmp_int %lhs, %rhs "eq" : i32, i32 -> i1
        ```
    }];
    
    let arguments = (ins I32:$lhs, I32:$rhs, StrAttr:$predicate);
    let results = (outs I1:$result);
    
    let assemblyFormat = "$lhs `,` $rhs $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Pg_CmpTextOp : Pg_Op<"cmp_text", [Pure]> {
    let summary = "Compare two text values";
    let description = [{
        Compares two text values with PostgreSQL semantics.
        
        Example:
        ```mlir
        %result = pg.cmp_text %lhs, %rhs "eq" : !pg.text, !pg.text -> i1
        ```
    }];
    
    let arguments = (ins Pg_TextType:$lhs, Pg_TextType:$rhs, StrAttr:$predicate);
    let results = (outs I1:$result);
    
    let assemblyFormat = "$lhs `,` $rhs $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Aggregation Operations
//===----------------------------------------------------------------------===//

def Pg_AggregateSumOp : Pg_Op<"aggregate_sum", [Pure]> {
    let summary = "Sum aggregation operation";
    let description = [{
        Performs sum aggregation over integer values, handling NULL values according
        to PostgreSQL semantics. Supports DISTINCT aggregation.
        
        Example:
        ```mlir
        %sum = pg.aggregate_sum %input_values {distinct = false} : !pg.table_handle -> i64
        %sum_distinct = pg.aggregate_sum %input_values {distinct = true} : !pg.table_handle -> i64
        ```
    }];
    
    let arguments = (ins Pg_TableHandleType:$input, DefaultValuedAttr<BoolAttr, "false">:$distinct);
    let results = (outs I64:$result);
    
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Pg_AggregateCountOp : Pg_Op<"aggregate_count", [Pure]> {
    let summary = "Count aggregation operation";
    let description = [{
        Counts the number of non-NULL values. Supports DISTINCT counting.
        
        Example:
        ```mlir
        %count = pg.aggregate_count %input_values {distinct = false} : !pg.table_handle -> i64
        %count_distinct = pg.aggregate_count %input_values {distinct = true} : !pg.table_handle -> i64
        ```
    }];
    
    let arguments = (ins Pg_TableHandleType:$input, DefaultValuedAttr<BoolAttr, "false">:$distinct);
    let results = (outs I64:$result);
    
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Output Operations
//===----------------------------------------------------------------------===//

def Pg_YieldTupleOp : Pg_Op<"yield_tuple", [Pure]> {
    let summary = "Yield tuple to query result";
    let description = [{
        Yields a tuple to the query result set. This represents the final
        output of a query. Supports ordering metadata for ORDER BY clauses.
        
        Example:
        ```mlir
        pg.yield_tuple %tuple {ordering = "asc"} : !pg.tuple<i32, !pg.text>
        pg.yield_tuple %tuple {ordering = "desc"} : !pg.tuple<i32, !pg.text>
        ```
    }];
    
    let arguments = (ins Pg_TupleType:$tuple, DefaultValuedAttr<StrAttr, "\"none\"">:$ordering);
    let results = (outs);
    
    let assemblyFormat = "$tuple attr-dict `:` type($tuple)";
}

#endif // PG_OPS