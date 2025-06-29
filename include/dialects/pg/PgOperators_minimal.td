//===- PgOperators.td - PostgreSQL Operator Definitions --*- tablegen -*-===//
//
// Minimal test version of PostgreSQL operators
//
//===----------------------------------------------------------------------===//

#ifndef PG_OPERATORS_TD
#define PG_OPERATORS_TD

include "dialects/pg/PgBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

#endif // PG_OPERATORS_TD