#ifndef PGX_DIALECT_DB_IR_OPS
#define PGX_DIALECT_DB_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td" 
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

//===----------------------------------------------------------------------===//
// DB dialect definition.
//===----------------------------------------------------------------------===//

def DB_Dialect : Dialect {
    let name = "db";
    let summary = "A dialect for Database operations in pgx-lower";
    let description = [{
        The DB dialect provides operations for PostgreSQL integration,
        handling table access, tuple iteration, field extraction, and result streaming.
        It serves as the interface between MLIR operations and PostgreSQL runtime.
    }];
    let cppNamespace = "::pgx::db";
    
    // TEMPORARY: Disable properties system for MLIR v20 compatibility
    // TODO: Remove this workaround and fix resulting compilation errors
    let usePropertiesForAttributes = 0;
    
    let extraClassDeclaration = [{
        private:
        void registerTypes();
        
        public:
            // Type parsing and printing methods for DB dialect custom types
            ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
            void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
    }];
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class DB_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name> {
    let mnemonic = typeMnemonic;
}

// Fixed nullable types for common SQL types
def DB_NullableI32 : DB_Type<"NullableI32", "nullable_i32"> {
    let summary = "nullable 32-bit integer for SQL semantics";
    let description = [{
        A nullable 32-bit integer type that can represent either an integer value
        or SQL NULL. This provides proper SQL NULL semantics for integer operations.
    }];
}

def DB_NullableI64 : DB_Type<"NullableI64", "nullable_i64"> {
    let summary = "nullable 64-bit integer for SQL semantics";  
    let description = [{
        A nullable 64-bit integer type that can represent either an integer value
        or SQL NULL. This provides proper SQL NULL semantics for integer operations.
    }];
}

def DB_NullableF64 : DB_Type<"NullableF64", "nullable_f64"> {
    let summary = "nullable 64-bit float for SQL semantics";
    let description = [{
        A nullable double-precision float type that can represent either a float value
        or SQL NULL. This provides proper SQL NULL semantics for float operations.
    }];
}

def DB_NullableBool : DB_Type<"NullableBool", "nullable_bool"> {
    let summary = "nullable boolean for SQL semantics";
    let description = [{
        A nullable boolean type that can represent true, false, or SQL NULL.
        This provides proper SQL three-valued logic semantics.
    }];
}

// Simple NULL marker type for creating NULL literals
def DB_SqlNull : DB_Type<"SqlNull", "sql_null"> {
    let summary = "SQL NULL literal type";
    let description = [{
        Type representing SQL NULL literals. Used by the NullOp operation
        to create NULL values that can be cast to nullable<T> types.
    }];
}

def DB_ExternalSource : DB_Type<"ExternalSource", "external_source"> {
    let summary = "PostgreSQL table handle";
    let description = [{
        Opaque handle representing a PostgreSQL table being scanned.
        Contains internal state for tuple iteration and field access.
    }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

// Core type constraints  
def DB_SqlNullType : AnyTypeOf<[DB_SqlNull]>;
def DB_NullableTypes : AnyTypeOf<[DB_NullableI32, DB_NullableI64, DB_NullableF64, DB_NullableBool]>;

// SQL data type constraints
def DB_SQLInteger : AnyTypeOf<[I8, I16, I32, I64]>;
def DB_SQLFloat : AnyTypeOf<[F32, F64]>; 
def DB_SQLNumeric : AnyTypeOf<[DB_SQLInteger, DB_SQLFloat]>;
def DB_SQLBasic : AnyTypeOf<[DB_SQLNumeric, I1]>;

// Nullable type constraints for SQL operations 
def DB_NullableInteger : AnyTypeOf<[DB_SQLInteger, DB_NullableI32, DB_NullableI64]>;
def DB_NullableFloat : AnyTypeOf<[DB_SQLFloat, DB_NullableF64]>;
def DB_NullableNumeric : AnyTypeOf<[DB_SQLNumeric, DB_NullableI32, DB_NullableI64, DB_NullableF64]>;
def DB_NullableLogical : AnyTypeOf<[I1, DB_NullableBool]>;
def DB_NullableValue : AnyTypeOf<[DB_SQLBasic, DB_NullableTypes, DB_SqlNull]>;

//===----------------------------------------------------------------------===//
// Base DB operation definition.
//===----------------------------------------------------------------------===//

class DB_Op<string mnemonic, list<Trait> traits = []> : Op<DB_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// PostgreSQL-specific Operations (6 total)
//===----------------------------------------------------------------------===//

def GetExternalOp : DB_Op<"get_external", [Pure]> {
    let summary = "initialize PostgreSQL table access";
    let description = [{
        Initialize access to a PostgreSQL table by opening the table relation
        and setting up scanning infrastructure. Returns an opaque handle for
        subsequent operations.
        
        Example:
        ```mlir
        %handle = db.get_external %table_oid : i64 -> !db.external_source
        ```
    }];
    
    let arguments = (ins I64:$table_oid);
    let results = (outs DB_ExternalSource:$handle);
}

def IterateExternalOp : DB_Op<"iterate_external", []> {
    let summary = "get next tuple from PostgreSQL table";
    let description = [{
        Advance the table scan to the next tuple. Returns true if a tuple
        is available, false if end of table reached.
        
        Example:
        ```mlir
        %has_tuple = db.iterate_external %handle : !db.external_source -> i1
        ```
    }];
    
    let arguments = (ins DB_ExternalSource:$handle);
    let results = (outs I1:$has_tuple);
}

def GetFieldOp : DB_Op<"get_field", []> {
    let summary = "extract field value from current tuple with null handling";
    let description = [{
        Extract a field value from the current tuple with proper SQL null handling.
        Returns a nullable value that can represent either the actual field value
        or NULL if the field is null in the database.
        
        Example:
        ```mlir
        %value = db.get_field %handle, %field_index, %type_oid : 
            !db.external_source, index, i32 -> !db.nullable<i32>
        ```
    }];
    
    let arguments = (ins 
        DB_ExternalSource:$handle,
        Index:$field_index, 
        I32:$type_oid
    );
    let results = (outs DB_NullableValue:$value);
}

def StoreResultOp : DB_Op<"store_result", []> {
    let summary = "store computed value for result tuple";
    let description = [{
        Store a computed value in the result tuple being constructed.
        Handles conversion from MLIR types to PostgreSQL Datums.
        Supports nullable values with proper NULL handling.
        
        Example:
        ```mlir
        db.store_result %value, %field_index : !db.nullable<i32>, index
        ```
    }];
    
    let arguments = (ins 
        DB_NullableValue:$value,
        Index:$field_index
    );
    let results = (outs);
}

def StreamResultsOp : DB_Op<"stream_results", []> {
    let summary = "output accumulated results to PostgreSQL";
    let description = [{
        Finalize and output the accumulated result tuples to PostgreSQL.
        This triggers the actual data transmission back to the client.
        
        Example:
        ```mlir
        db.stream_results
        ```
    }];
    
    let arguments = (ins);
    let results = (outs);
}

def GetColumnOp : DB_Op<"get_column", [Pure]> {
    let summary = "get column value for RelAlg compatibility";
    let description = [{
        Extract a column value for compatibility with RelAlg lowering patterns.
        This operation provides an interface compatible with RelAlg GetColumnOp
        while working with the PostgreSQL runtime system.
        
        Example:
        ```mlir
        %value = db.get_column %handle, %column_attr : 
            !db.external_source, !relalg.columnref -> !db.nullable<i32>
        ```
    }];
    
    let arguments = (ins 
        DB_ExternalSource:$handle,
        AnyAttr:$column_attr
    );
    let results = (outs DB_NullableValue:$result);
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations (5 total) - Reused from LingoDB
//===----------------------------------------------------------------------===//

def AddOp : DB_Op<"add", [Pure, Commutative]> {
    let summary = "addition operation with null handling";
    let description = [{
        Performs addition of two values with SQL null semantics.
        If either operand is null, the result is null.
        Supports nullable numeric types.
    }];
    
    let arguments = (ins DB_NullableNumeric:$lhs, DB_NullableNumeric:$rhs);
    let results = (outs DB_NullableNumeric:$result);
}

def SubOp : DB_Op<"sub", [Pure]> {
    let summary = "subtraction operation with null handling";
    let description = [{
        Performs subtraction of two values with SQL null semantics.
        If either operand is null, the result is null.
        Supports nullable numeric types.
    }];
    
    let arguments = (ins DB_NullableNumeric:$lhs, DB_NullableNumeric:$rhs);
    let results = (outs DB_NullableNumeric:$result);
}

def MulOp : DB_Op<"mul", [Pure, Commutative]> {
    let summary = "multiplication operation with null handling";
    let description = [{
        Performs multiplication of two values with SQL null semantics.
        If either operand is null, the result is null.
        Supports nullable numeric types.
    }];
    
    let arguments = (ins DB_NullableNumeric:$lhs, DB_NullableNumeric:$rhs);
    let results = (outs DB_NullableNumeric:$result);
}

def DivOp : DB_Op<"div", [Pure]> {
    let summary = "division operation with null handling";
    let description = [{
        Performs division of two values with SQL null semantics.
        If either operand is null, the result is null.
        Division by zero behavior follows PostgreSQL semantics.
        Supports nullable numeric types.
    }];
    
    let arguments = (ins DB_NullableNumeric:$lhs, DB_NullableNumeric:$rhs);
    let results = (outs DB_NullableNumeric:$result);
}

def ModOp : DB_Op<"mod", [Pure]> {
    let summary = "modulo operation with null handling";
    let description = [{
        Performs modulo operation with SQL null semantics.
        If either operand is null, the result is null.
        Supports nullable numeric types.
    }];
    
    let arguments = (ins DB_NullableNumeric:$lhs, DB_NullableNumeric:$rhs);
    let results = (outs DB_NullableNumeric:$result);
}

//===----------------------------------------------------------------------===//
// Logical Operations (3 total) - Reused from LingoDB
//===----------------------------------------------------------------------===//

def AndOp : DB_Op<"and", [Pure, Commutative]> {
    let summary = "logical AND with three-valued logic";
    let description = [{
        Performs logical AND with SQL three-valued logic (true, false, null).
        Follows SQL standard null propagation rules.
        Supports nullable boolean types.
    }];
    
    let arguments = (ins DB_NullableLogical:$lhs, DB_NullableLogical:$rhs);
    let results = (outs DB_NullableLogical:$result);
}

def OrOp : DB_Op<"or", [Pure, Commutative]> {
    let summary = "logical OR with three-valued logic";
    let description = [{
        Performs logical OR with SQL three-valued logic (true, false, null).
        Follows SQL standard null propagation rules.
        Supports nullable boolean types.
    }];
    
    let arguments = (ins DB_NullableLogical:$lhs, DB_NullableLogical:$rhs);
    let results = (outs DB_NullableLogical:$result);
}

def NotOp : DB_Op<"not", [Pure]> {
    let summary = "logical NOT with three-valued logic";
    let description = [{
        Performs logical NOT with SQL three-valued logic.
        NOT NULL = NULL in SQL semantics.
        Supports nullable boolean types.
    }];
    
    let arguments = (ins DB_NullableLogical:$operand);
    let results = (outs DB_NullableLogical:$result);
}

//===----------------------------------------------------------------------===//
// Comparison Operations (1 total) - Reused from LingoDB  
//===----------------------------------------------------------------------===//

// Enum for comparison predicates
def DB_CmpPredicate : I32EnumAttr<"CmpPredicate",
    "Comparison predicate", [
        I32EnumAttrCase<"eq", 0, "eq">,
        I32EnumAttrCase<"ne", 1, "ne">, 
        I32EnumAttrCase<"lt", 2, "lt">,
        I32EnumAttrCase<"le", 3, "le">,
        I32EnumAttrCase<"gt", 4, "gt">,
        I32EnumAttrCase<"ge", 5, "ge">
    ]> {
    let cppNamespace = "::pgx::db";
    let genSpecializedAttr = 0;
}

def CompareOp : DB_Op<"compare", [Pure]> {
    let summary = "comparison operation with null handling";
    let description = [{
        Performs comparison between two values with SQL null semantics.
        Any comparison involving null returns null (unknown).
        Supports nullable types with proper SQL three-valued logic.
    }];
    
    let arguments = (ins 
        I32Attr:$predicate,
        DB_NullableValue:$lhs, 
        DB_NullableValue:$rhs
    );
    let results = (outs DB_NullableLogical:$result);
}

//===----------------------------------------------------------------------===//
// Nullable Operations (4 total) - Reused from LingoDB
//===----------------------------------------------------------------------===//

def NullOp : DB_Op<"null", [Pure]> {
    let summary = "create SQL NULL value";
    let description = [{
        Creates a SQL NULL marker value for use in SQL three-valued logic.
        
        Example:
        ```mlir
        %null = db.null : !db.sql_null
        ```
    }];
    
    let arguments = (ins);
    let results = (outs DB_SqlNull:$result);
}

def AsNullableOp : DB_Op<"as_nullable", [Pure]> {
    let summary = "convert value to nullable type";
    let description = [{
        Converts a non-nullable value to a nullable type for SQL semantics.
        The result is a nullable version of the input type that can represent
        either the original value or NULL.
        
        Example:
        ```mlir
        %nullable_val = db.as_nullable %input : i32 -> !db.nullable<i32>
        ```
    }];
    
    let arguments = (ins DB_SQLBasic:$value);
    let results = (outs DB_NullableValue:$result);
}

def IsNullOp : DB_Op<"isnull", [Pure]> {
    let summary = "test if nullable value is null";
    let description = [{
        Tests whether a nullable value represents SQL NULL.
        Returns true if the value is NULL, false if it contains a value.
        
        Example:
        ```mlir
        %is_null = db.isnull %value : !db.nullable<i32> -> i1
        ```
    }];
    
    let arguments = (ins DB_NullableValue:$value);
    let results = (outs I1:$result);
}

def NullableGetValOp : DB_Op<"nullable_get_val", [Pure]> {
    let summary = "extract value from nullable type";
    let description = [{
        Extracts the underlying value from a nullable type.
        Behavior is undefined if the nullable value is NULL.
        Must be guarded by IsNullOp check in safe code.
        
        Example:
        ```mlir
        %result = db.nullable_get_val %value : !db.nullable<i32> -> i32
        ```
    }];
    
    let arguments = (ins DB_NullableValue:$value);
    let results = (outs DB_SQLBasic:$result);
}

//===----------------------------------------------------------------------===//
// Utility Operations (2 total) - Reused from LingoDB
//===----------------------------------------------------------------------===//

def ConstantOp : DB_Op<"constant", [Pure]> {
    let summary = "create constant value";
    let description = [{
        Creates a constant value of the specified type.
        Can create both nullable and non-nullable constants.
        
        Example:
        ```mlir
        %const = db.constant 42 : i32
        %nullable_const = db.constant 42 : !db.nullable<i32>
        ```
    }];
    
    let arguments = (ins AnyAttr:$value);
    let results = (outs DB_NullableValue:$result);
}

def CastOp : DB_Op<"cast", [Pure]> {
    let summary = "cast value to different type";
    let description = [{
        Performs type casting with SQL semantics.
        Handles nullable values appropriately - NULL remains NULL.
        Supports casting between nullable and non-nullable types.
        
        Example:
        ```mlir
        %cast = db.cast %value : !db.nullable<i32> to !db.nullable<i64>
        ```
    }];
    
    let arguments = (ins DB_NullableValue:$input);
    let results = (outs DB_NullableValue:$result);
}

#endif // PGX_DIALECT_DB_IR_OPS