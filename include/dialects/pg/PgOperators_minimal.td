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

#endif // PG_OPERATORS_TD