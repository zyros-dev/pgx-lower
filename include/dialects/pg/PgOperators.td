//===- PgOperators.td - PostgreSQL Operator Definitions --*- tablegen -*-===//
//
// Minimal test version of PostgreSQL operators
//
//===----------------------------------------------------------------------===//

#ifndef PG_OPERATORS_TD
#define PG_OPERATORS_TD

include "dialects/pg/PgBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Simple Test Operators - Building up gradually
//===----------------------------------------------------------------------===//

def PgAddOp : Pg_Op<"add", [Pure]> {
  let summary = "PostgreSQL addition operation";
  let description = [{
    Simple addition test operation.
  }];
  
  let arguments = (ins I32:$left, I32:$right);
  let results = (outs I32:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

def PgSubOp : Pg_Op<"sub", [Pure]> {
  let summary = "PostgreSQL subtraction operation";
  let description = [{
    Simple subtraction test operation.
  }];
  
  let arguments = (ins I32:$left, I32:$right);
  let results = (outs I32:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

def PgMulOp : Pg_Op<"mul", [Pure]> {
  let summary = "PostgreSQL multiplication operation";
  let description = [{
    Simple multiplication test operation.
  }];
  
  let arguments = (ins I32:$left, I32:$right);
  let results = (outs I32:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

def PgDivOp : Pg_Op<"div", [Pure]> {
  let summary = "PostgreSQL division operation";
  let description = [{
    Simple division test operation.
  }];
  
  let arguments = (ins I32:$left, I32:$right);
  let results = (outs I32:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

def PgModOp : Pg_Op<"mod", [Pure]> {
  let summary = "PostgreSQL modulo operation";
  let description = [{
    Simple modulo test operation.
  }];
  
  let arguments = (ins I32:$left, I32:$right);
  let results = (outs I32:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

// Test comparison operators with integer predicate attribute
def PgCmpOp : Pg_Op<"compare", [Pure]> {
  let summary = "PostgreSQL comparison operation";
  let description = [{
    Simple comparison test operation with integer predicate.
  }];
  
  let arguments = (ins 
    I32Attr:$predicate,
    I32:$left, 
    I32:$right
  );
  let results = (outs I1:$result);
  
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
}

// Test logical operators - next level of complexity
def PgAndOp : Pg_Op<"and", [Pure]> {
  let summary = "PostgreSQL logical AND operation";
  let description = [{
    Simple logical AND test operation.
  }];
  
  let arguments = (ins I1:$left, I1:$right);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

def PgOrOp : Pg_Op<"or", [Pure]> {
  let summary = "PostgreSQL logical OR operation";
  let description = [{
    Simple logical OR test operation.
  }];
  
  let arguments = (ins I1:$left, I1:$right);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

def PgNotOp : Pg_Op<"not", [Pure]> {
  let summary = "PostgreSQL logical NOT operation";
  let description = [{
    Simple logical NOT test operation.
  }];
  
  let arguments = (ins I1:$operand);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

// Test with custom PostgreSQL types - potential source of segfault
def PgTextAddOp : Pg_Op<"text_add", [Pure]> {
  let summary = "PostgreSQL text addition operation";
  let description = [{
    Test operation using custom PostgreSQL text type.
  }];
  
  let arguments = (ins Pg_TextType:$left, Pg_TextType:$right);
  let results = (outs Pg_TextType:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

// Test variadic arguments - potential segfault source
def PgConcatOp : Pg_Op<"concat", [Pure]> {
  let summary = "PostgreSQL concatenation with variadic arguments";
  let description = [{
    Test operation using variadic arguments.
  }];
  
  let arguments = (ins Variadic<Pg_TextType>:$inputs);
  let results = (outs Pg_TextType:$result);
  
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

// Test type constraints - potential segfault source
def PgGenericEqOp : Pg_Op<"generic_eq", [Pure]> {
  let summary = "PostgreSQL generic equality operation";
  let description = [{
    Test operation using type constraints and generic types.
  }];
  
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs I1:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left) `,` type($right)";
}

// Test optional arguments - potential segfault source
def PgCoalesceOp : Pg_Op<"coalesce", [Pure]> {
  let summary = "PostgreSQL coalesce operation";
  let description = [{
    Test operation using optional arguments.
  }];
  
  let arguments = (ins AnyType:$value, Optional<AnyType>:$default_value);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$value (`,` $default_value^ `:` type($default_value))? attr-dict `:` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Comprehensive TPC-H Operators - Using Proven Working Patterns
//===----------------------------------------------------------------------===//

// Numeric comparison operators using proven comparison pattern
def PgNeOp : Pg_Op<"ne", [Pure]> {
  let summary = "PostgreSQL not equal operation";
  let arguments = (ins I32Attr:$predicate, AnyType:$left, AnyType:$right);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
}

def PgLtOp : Pg_Op<"lt", [Pure]> {
  let summary = "PostgreSQL less than operation";
  let arguments = (ins I32Attr:$predicate, AnyType:$left, AnyType:$right);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
}

def PgLeOp : Pg_Op<"le", [Pure]> {
  let summary = "PostgreSQL less than or equal operation";
  let arguments = (ins I32Attr:$predicate, AnyType:$left, AnyType:$right);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
}

def PgGtOp : Pg_Op<"gt", [Pure]> {
  let summary = "PostgreSQL greater than operation";
  let arguments = (ins I32Attr:$predicate, AnyType:$left, AnyType:$right);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
}

def PgGeOp : Pg_Op<"ge", [Pure]> {
  let summary = "PostgreSQL greater than or equal operation";
  let arguments = (ins I32Attr:$predicate, AnyType:$left, AnyType:$right);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
}

// Aggregate functions using variadic pattern
def PgSumOp : Pg_Op<"sum", [Pure]> {
  let summary = "PostgreSQL SUM aggregate operation";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def PgAvgOp : Pg_Op<"avg", [Pure]> {
  let summary = "PostgreSQL AVG aggregate operation";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def PgCountOp : Pg_Op<"count", [Pure]> {
  let summary = "PostgreSQL COUNT aggregate operation";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs I64:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def PgMinOp : Pg_Op<"min", [Pure]> {
  let summary = "PostgreSQL MIN aggregate operation";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def PgMaxOp : Pg_Op<"max", [Pure]> {
  let summary = "PostgreSQL MAX aggregate operation";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

// NULL handling using Optional pattern
def PgIsNullOp : Pg_Op<"is_null", [Pure]> {
  let summary = "PostgreSQL IS NULL operation";
  let arguments = (ins AnyType:$operand);
  let results = (outs I1:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def PgIsNotNullOp : Pg_Op<"is_not_null", [Pure]> {
  let summary = "PostgreSQL IS NOT NULL operation";
  let arguments = (ins AnyType:$operand);
  let results = (outs I1:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

// String operations using PostgreSQL types
def PgLikeOp : Pg_Op<"like", [Pure]> {
  let summary = "PostgreSQL LIKE operation";
  let arguments = (ins Pg_TextType:$text, Pg_TextType:$pattern);
  let results = (outs I1:$result);
  let assemblyFormat = "$text `,` $pattern attr-dict `:` type($text)";
}

def PgSubstringOp : Pg_Op<"substring", [Pure]> {
  let summary = "PostgreSQL SUBSTRING operation";
  let arguments = (ins Pg_TextType:$text, I32:$start, Optional<I32>:$length);
  let results = (outs Pg_TextType:$result);
  let assemblyFormat = "$text `,` $start (`,` $length^)? attr-dict `:` type($text)";
}

#endif // PG_OPERATORS_TD