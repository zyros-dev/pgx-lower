//===- DBDialect.td - Database dialect definition --------*- tablegen -*-===//
//
// Based on LingoDB's DB dialect
// Imperative database dialect with SQL semantics and NULL support
//
//===----------------------------------------------------------------------===//

#ifndef DB_DIALECT
#define DB_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Database dialect definition
//===----------------------------------------------------------------------===//

def DB_Dialect : Dialect {
    let name = "db";
    let summary = "Imperative database dialect with SQL semantics";
    let description = [{
        The DB dialect provides type-polymorphic database operations with
        proper NULL handling following SQL semantics. This is a simplified
        version of LingoDB's DB dialect tailored for pgx-lower.
    }];
    let cppNamespace = "::pgx_lower::compiler::dialect::db";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Database types
//===----------------------------------------------------------------------===//

class DB_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<DB_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// Nullable type - wraps any type to make it nullable
def DB_NullableType : DB_Type<"Nullable", "nullable"> {
    let summary = "Nullable type wrapper";
    let description = [{
        Wraps any type to indicate it can be NULL. This follows SQL semantics
        where any value can potentially be NULL.
    }];
    let parameters = (ins "mlir::Type":$valueType);
    let assemblyFormat = "`<` $valueType `>`";
}

// Forward declarations for types defined in DBOps.td
def DB_DateType : DB_Type<"Date", "date">;
def DB_TimestampType : DB_Type<"Timestamp", "timestamp">;
def DB_IntervalType : DB_Type<"Interval", "interval">;

// Decimal type with precision and scale
def DB_DecimalType : DB_Type<"Decimal", "decimal"> {
    let summary = "SQL DECIMAL type";
    let description = [{
        Represents a SQL DECIMAL value with specified precision and scale.
    }];
    let parameters = (ins "unsigned":$precision, "unsigned":$scale);
    let assemblyFormat = "`<` $precision `,` $scale `>`";
}

// String type
def DB_StringType : DB_Type<"String", "string"> {
    let summary = "SQL VARCHAR/TEXT type";
    let description = [{
        Represents a variable-length string value.
    }];
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// Any type that can be nullable
def NullableDBType : AnyTypeOf<[AnyType, DB_NullableType]>;

// Numeric types for arithmetic operations
def DB_NumericType : AnyTypeOf<[I8, I16, I32, I64, F32, F64, DB_DecimalType]>;

// Types that support comparison
def DB_ComparableType : AnyTypeOf<[DB_NumericType, DB_StringType, DB_DateType, DB_TimestampType]>;

//===----------------------------------------------------------------------===//
// Database operation base classes
//===----------------------------------------------------------------------===//

class DB_Op<string mnemonic, list<Trait> traits = []> :
    Op<DB_Dialect, mnemonic, traits>;

// Base class for binary operations (add, sub, mul, div, mod)
class DB_BinaryOp<string mnemonic, list<Trait> traits = []> :
    DB_Op<mnemonic, !listconcat(traits, [Pure, InferTypeOpInterface])> {
    let arguments = (ins NullableDBType:$left, NullableDBType:$right);
    let results = (outs AnyType:$result);
    
    let assemblyFormat = [{
        $left `,` $right attr-dict `:` type($left) `,` type($right) `->` type($result)
    }];
    
    let extraClassDeclaration = [{
        static mlir::LogicalResult inferReturnTypes(
            mlir::MLIRContext *context,
            std::optional<mlir::Location> location,
            mlir::ValueRange operands,
            mlir::DictionaryAttr attributes,
            mlir::OpaqueProperties properties,
            mlir::RegionRange regions,
            llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes);
    }];
}

//===----------------------------------------------------------------------===//
// Database operations
//===----------------------------------------------------------------------===//

// Constant operation
def DB_ConstantOp : DB_Op<"constant", [Pure, ConstantLike]> {
    let summary = "Database constant";
    let description = [{
        Creates a constant value of any database type.
    }];
    
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);
    
    let builders = [
        OpBuilder<(ins "mlir::Attribute":$value, "mlir::Type":$type), [{
            build($_builder, $_state, type, value);
        }]>
    ];
    
    let hasFolder = 1;
    let hasCustomAssemblyFormat = 1;
}

// NULL constant
def DB_NullOp : DB_Op<"null", [Pure, ConstantLike]> {
    let summary = "NULL constant";
    let description = [{
        Creates a NULL value of the specified nullable type.
    }];
    
    let results = (outs DB_NullableType:$result);
    
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

// Arithmetic operations
def DB_AddOp : DB_BinaryOp<"add"> {
    let summary = "Addition with NULL handling";
    let description = [{
        Performs addition following SQL semantics. If either operand is NULL,
        the result is NULL.
    }];
}

def DB_SubOp : DB_BinaryOp<"sub"> {
    let summary = "Subtraction with NULL handling";
    let description = [{
        Performs subtraction following SQL semantics. If either operand is NULL,
        the result is NULL.
    }];
}

def DB_MulOp : DB_BinaryOp<"mul"> {
    let summary = "Multiplication with NULL handling";
    let description = [{
        Performs multiplication following SQL semantics. If either operand is NULL,
        the result is NULL.
    }];
}

def DB_DivOp : DB_BinaryOp<"div"> {
    let summary = "Division with NULL handling";
    let description = [{
        Performs division following SQL semantics. If either operand is NULL,
        the result is NULL. Division by zero produces NULL.
    }];
}

def DB_ModOp : DB_BinaryOp<"mod"> {
    let summary = "Modulo with NULL handling";
    let description = [{
        Performs modulo operation following SQL semantics. If either operand is NULL,
        the result is NULL.
    }];
}

// Comparison operation
def DB_CompareOp : DB_Op<"compare", [Pure]> {
    let summary = "Comparison with NULL handling";
    let description = [{
        Performs comparison following SQL three-valued logic. Comparisons
        involving NULL produce NULL (unknown) rather than true/false.
    }];
    
    let arguments = (ins 
        NullableDBType:$left,
        NullableDBType:$right,
        I64Attr:$predicate
    );
    let results = (outs NullableDBType:$result);
    
    let assemblyFormat = [{
        $predicate $left `,` $right attr-dict `:` type($left) `,` type($right) `->` type($result)
    }];
}

// NULL checking operations
def DB_IsNullOp : DB_Op<"is_null", [Pure]> {
    let summary = "Check if value is NULL";
    let description = [{
        Returns true if the value is NULL, false otherwise.
        This always returns a non-nullable boolean.
    }];
    
    let arguments = (ins NullableDBType:$value);
    let results = (outs I1:$result);
    
    let assemblyFormat = [{
        $value attr-dict `:` type($value)
    }];
}

def DB_IsNotNullOp : DB_Op<"is_not_null", [Pure]> {
    let summary = "Check if value is not NULL";
    let description = [{
        Returns true if the value is not NULL, false otherwise.
        This always returns a non-nullable boolean.
    }];
    
    let arguments = (ins NullableDBType:$value);
    let results = (outs I1:$result);
    
    let assemblyFormat = [{
        $value attr-dict `:` type($value)
    }];
}

// COALESCE operation
def DB_CoalesceOp : DB_Op<"coalesce", [Pure]> {
    let summary = "COALESCE - return first non-NULL value";
    let description = [{
        Returns the first non-NULL value from the operands. If all operands
        are NULL, returns NULL.
    }];
    
    let arguments = (ins Variadic<NullableDBType>:$values);
    let results = (outs NullableDBType:$result);
    
    let assemblyFormat = [{
        `(` $values `)` attr-dict `:` type($values) `->` type($result)
    }];
}

// Logical operations with three-valued logic
def DB_AndOp : DB_Op<"and", [Pure]> {
    let summary = "Logical AND with three-valued logic";
    let description = [{
        Performs logical AND following SQL three-valued logic:
        - true AND true = true
        - true AND false = false
        - true AND NULL = NULL
        - false AND anything = false
        - NULL AND NULL = NULL
    }];
    
    let arguments = (ins NullableDBType:$left, NullableDBType:$right);
    let results = (outs NullableDBType:$result);
    
    let assemblyFormat = [{
        $left `,` $right attr-dict `:` type($left) `,` type($right) `->` type($result)
    }];
}

def DB_OrOp : DB_Op<"or", [Pure]> {
    let summary = "Logical OR with three-valued logic";
    let description = [{
        Performs logical OR following SQL three-valued logic:
        - true OR anything = true
        - false OR false = false
        - false OR true = true
        - false OR NULL = NULL
        - NULL OR NULL = NULL
    }];
    
    let arguments = (ins NullableDBType:$left, NullableDBType:$right);
    let results = (outs NullableDBType:$result);
    
    let assemblyFormat = [{
        $left `,` $right attr-dict `:` type($left) `,` type($right) `->` type($result)
    }];
}

def DB_NotOp : DB_Op<"not", [Pure]> {
    let summary = "Logical NOT with three-valued logic";
    let description = [{
        Performs logical NOT following SQL three-valued logic:
        - NOT true = false
        - NOT false = true
        - NOT NULL = NULL
    }];
    
    let arguments = (ins NullableDBType:$value);
    let results = (outs NullableDBType:$result);
    
    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($result)
    }];
}

// Type conversion operations
def DB_AsNullableOp : DB_Op<"as_nullable", [Pure]> {
    let summary = "Convert value to nullable type";
    let description = [{
        Wraps a non-nullable value in a nullable type.
    }];
    
    let arguments = (ins AnyType:$value);
    let results = (outs DB_NullableType:$result);
    
    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($result)
    }];
}

def DB_NullableGetValOp : DB_Op<"nullable_get_val", [Pure]> {
    let summary = "Extract value from nullable type";
    let description = [{
        Extracts the underlying value from a nullable type. The behavior
        is undefined if the value is NULL - caller must check first.
    }];
    
    let arguments = (ins DB_NullableType:$nullable);
    let results = (outs AnyType:$value);
    
    let assemblyFormat = [{
        $nullable attr-dict `:` type($nullable) `->` type($value)
    }];
}

#endif // DB_DIALECT