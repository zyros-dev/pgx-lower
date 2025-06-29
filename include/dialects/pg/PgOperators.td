//===- PgOperators.td - PostgreSQL Operator Definitions --*- tablegen -*-===//
//
// This file defines PostgreSQL operator operations for the pg dialect.
// Based on LingoDB patterns but adapted for PostgreSQL integration.
//
//===----------------------------------------------------------------------===//

#ifndef PG_OPERATORS_TD
#define PG_OPERATORS_TD

include "dialects/pg/PgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Base Operator Classes
//===----------------------------------------------------------------------===//

// Base class for binary operators (arithmetic, comparison)
class PgBinaryOp<string mnemonic, list<Trait> traits = []> : 
    Pg_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Pg_AnyType:$left, Pg_AnyType:$right);
  let results = (outs Pg_AnyType:$result);
  
  let assemblyFormat = "$left `,` $right attr-dict `:` type($left)";
}

// Base class for logical operators  
class PgLogicalOp<string mnemonic, list<Trait> traits = []> :
    Pg_Op<mnemonic, traits # [Pure]> {
  let results = (outs Pg_BooleanType:$result);
  let assemblyFormat = "$operands attr-dict `:` type($operands)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operators
//===----------------------------------------------------------------------===//

def PgAddOp : PgBinaryOp<"add"> {
  let summary = "PostgreSQL addition operation";
  let description = [{
    Performs addition of two PostgreSQL values with null handling.
    Follows PostgreSQL semantics where null + anything = null.
    
    Example:
    ```mlir
    %result = pg.add %lhs, %rhs : !pg.int4
    ```
  }];
}

def PgSubOp : PgBinaryOp<"sub"> {
  let summary = "PostgreSQL subtraction operation";
  let description = [{
    Performs subtraction of two PostgreSQL values with null handling.
    
    Example:
    ```mlir
    %result = pg.sub %lhs, %rhs : !pg.int4
    ```
  }];
}

def PgMulOp : PgBinaryOp<"mul"> {
  let summary = "PostgreSQL multiplication operation";
  let description = [{
    Performs multiplication of two PostgreSQL values with null handling.
    
    Example:
    ```mlir
    %result = pg.mul %lhs, %rhs : !pg.int4
    ```
  }];
}

def PgDivOp : PgBinaryOp<"div"> {
  let summary = "PostgreSQL division operation";
  let description = [{
    Performs division of two PostgreSQL values with null handling.
    Handles division by zero according to PostgreSQL semantics.
    
    Example:
    ```mlir
    %result = pg.div %lhs, %rhs : !pg.int4
    ```
  }];
}

def PgModOp : PgBinaryOp<"mod"> {
  let summary = "PostgreSQL modulo operation";
  let description = [{
    Performs modulo operation of two PostgreSQL values with null handling.
    
    Example:
    ```mlir
    %result = pg.mod %lhs, %rhs : !pg.int4
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Comparison Operators
//===----------------------------------------------------------------------===//

// Simplified comparison predicate (will enhance later)
// 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge

def PgCmpOp : Pg_Op<"compare", [Pure]> {
  let summary = "PostgreSQL comparison operation";
  let description = [{
    Performs comparison of two PostgreSQL values with specified predicate.
    Handles PostgreSQL's three-valued logic (true/false/null).
    
    Example:
    ```mlir
    %result = pg.compare eq %lhs, %rhs : !pg.int4
    ```
  }];
  
  let arguments = (ins 
    I32Attr:$predicate,
    Pg_AnyType:$left, 
    Pg_AnyType:$right
  );
  let results = (outs Pg_BooleanType:$result);
  
  let assemblyFormat = "$predicate $left `,` $right attr-dict `:` type($left) `,` type($right)";
  
  // TODO: Add helper methods for predicate checking once enum generation is fixed
}

//===----------------------------------------------------------------------===//
// Logical Operators
//===----------------------------------------------------------------------===//

def PgAndOp : PgLogicalOp<"and"> {
  let summary = "PostgreSQL logical AND operation";
  let description = [{
    Performs logical AND with PostgreSQL three-valued logic.
    Supports variadic number of operands.
    
    Truth table:
    - true AND true = true
    - true AND false = false  
    - true AND null = null
    - false AND anything = false
    - null AND true = null
    - null AND false = false
    - null AND null = null
    
    Example:
    ```mlir
    %result = pg.and %a, %b, %c : !pg.bool, !pg.bool, !pg.bool
    ```
  }];
  
  let arguments = (ins Variadic<Pg_BooleanType>:$operands);
  let assemblyFormat = "$operands attr-dict `:` type($operands)";
}

def PgOrOp : PgLogicalOp<"or"> {
  let summary = "PostgreSQL logical OR operation";
  let description = [{
    Performs logical OR with PostgreSQL three-valued logic.
    Supports variadic number of operands.
    
    Truth table:
    - true OR anything = true
    - false OR false = false
    - false OR true = true
    - false OR null = null
    - null OR true = true
    - null OR false = null
    - null OR null = null
    
    Example:
    ```mlir
    %result = pg.or %a, %b : !pg.bool, !pg.bool
    ```
  }];
  
  let arguments = (ins Variadic<Pg_BooleanType>:$operands);
  let assemblyFormat = "$operands attr-dict `:` type($operands)";
}

def PgNotOp : PgLogicalOp<"not"> {
  let summary = "PostgreSQL logical NOT operation";
  let description = [{
    Performs logical NOT with PostgreSQL three-valued logic.
    
    Truth table:
    - NOT true = false
    - NOT false = true
    - NOT null = null
    
    Example:
    ```mlir
    %result = pg.not %operand : !pg.bool
    ```
  }];
  
  let arguments = (ins Pg_BooleanType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

//===----------------------------------------------------------------------===//
// Null Handling Operators
//===----------------------------------------------------------------------===//

def PgIsNullOp : Pg_Op<"is_null", [Pure]> {
  let summary = "PostgreSQL IS NULL operation";
  let description = [{
    Tests if a value is null. Always returns a non-null boolean.
    
    Example:
    ```mlir
    %result = pg.is_null %value : !pg.int4
    ```
  }];
  
  let arguments = (ins Pg_AnyType:$operand);
  let results = (outs I1:$result);  // Always non-null boolean
  
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def PgIsNotNullOp : Pg_Op<"is_not_null", [Pure]> {
  let summary = "PostgreSQL IS NOT NULL operation";
  let description = [{
    Tests if a value is not null. Always returns a non-null boolean.
    Equivalent to NOT (IS NULL) but more efficient.
    
    Example:
    ```mlir
    %result = pg.is_not_null %value : !pg.int4
    ```
  }];
  
  let arguments = (ins Pg_AnyType:$operand);
  let results = (outs I1:$result);  // Always non-null boolean
  
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

#endif // PG_OPERATORS_TD