#ifndef PGX_DIALECT_DSA_IR_OPS
#define PGX_DIALECT_DSA_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td" 
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// DSA dialect definition.
//===----------------------------------------------------------------------===//

def DSA_Dialect : Dialect {
    let name = "dsa";
    let summary = "A dialect for Data Structure Abstraction in pgx-lower";
    let cppNamespace = "::pgx::mlir::dsa";
    
    // REQUIRED: Properties system disabled due to MLIR v20 compatibility issues
    // Removing this causes compilation errors with BytecodeOpInterface and Attribute resolution
    let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class DSA_Type<string name, string typeMnemonic> : TypeDef<DSA_Dialect, name> {
    let mnemonic = typeMnemonic;
}

// Base class for collection types following LingoDB patterns
class DSA_Collection<string name, string typeMnemonic> : TypeDef<DSA_Dialect, name, [], "::mlir::Type"> {
    let mnemonic = typeMnemonic;
}

def DSA_GenericIterable : DSA_Type<"GenericIterable", "generic_iterable"> {
    let summary = "generic iterable type returned by scan_source";
    let assemblyFormat = "";
}

def DSA_RecordBatch : DSA_Type<"RecordBatch", "record_batch"> {
    let summary = "chunk of records processed together";
    let assemblyFormat = "";
}

def DSA_Record : DSA_Type<"Record", "record"> {
    let summary = "individual row of data";
    let assemblyFormat = "";
}

def DSA_TableBuilder : DSA_Type<"TableBuilder", "table_builder"> {
    let summary = "result construction utility";
    let assemblyFormat = "";
}

def DSA_Table : DSA_Type<"Table", "table"> {
    let summary = "final materialized result table";
    let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Base DSA operation definition.
//===----------------------------------------------------------------------===//

class DSA_Op<string mnemonic, list<Trait> traits = []> : Op<DSA_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Data Access Operations
//===----------------------------------------------------------------------===//

def ScanSourceOp : DSA_Op<"scan_source", [Pure]> {
    let summary = "create iterator from table JSON description";
    let description = [{
        Creates an iterator from table JSON description.
        This is the entry point for data access in DSA.
        
        Example:
        ```mlir
        %iter = dsa.scan_source %json_desc : !dsa.generic_iterable
        ```
    }];

    let arguments = (ins StrAttr:$table_description);
    let results = (outs DSA_GenericIterable:$result);
    let assemblyFormat = "$table_description attr-dict `:` type($result)";
}

def AtOp : DSA_Op<"at", [Pure]> {
    let summary = "column value extraction with null handling";
    let description = [{
        Extracts a column value from a record with proper null handling.
        
        Example:
        ```mlir
        %value = dsa.at %record["column_name"] : (!dsa.record) -> i32
        ```
    }];

    let arguments = (ins DSA_Record:$record, StrAttr:$column_name);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$record `[` $column_name `]` attr-dict `:` `(` type($record) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def ForOp : DSA_Op<"for", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects
]> {
    let summary = "loop operation with implicit yield terminator";
    let description = [{
        Loop operation that iterates over an iterable with implicit yield terminator.
        The body region contains the loop body that processes each element.
        
        Example:
        ```mlir
        dsa.for %record in %iterable : !dsa.generic_iterable {
            // Process %record
            dsa.yield
        }
        ```
    }];

    let arguments = (ins DSA_GenericIterable:$iterable);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$iterable $body attr-dict `:` type($iterable)";
}

def YieldOp : DSA_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "loop terminator";
    let description = [{
        Terminator for DSA loop operations. Yields control back to the loop.
    }];

    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

//===----------------------------------------------------------------------===//
// Result Building Operations
//===----------------------------------------------------------------------===//

def CreateDSOp : DSA_Op<"create_ds", [Pure]> {
    let summary = "initialize result builder";
    let description = [{
        Initialize a result builder for constructing the final table.
        
        Example:
        ```mlir
        %builder = dsa.create_ds : !dsa.table_builder
        ```
    }];

    let arguments = (ins);
    let results = (outs DSA_TableBuilder:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def DSAppendOp : DSA_Op<"ds_append", []> {
    let summary = "append values to result";
    let description = [{
        Append values to the result builder.
        
        Example:
        ```mlir
        dsa.ds_append %builder, %value1, %value2 : !dsa.table_builder, i32, i32
        ```
    }];

    let arguments = (ins DSA_TableBuilder:$builder, Variadic<AnyType>:$values);
    let results = (outs);
    let assemblyFormat = "$builder `,` $values attr-dict `:` type($builder) (`,` type($values)^)?";
}

def NextRowOp : DSA_Op<"next_row", []> {
    let summary = "finalize row in builder";
    let description = [{
        Finalize the current row in the result builder and prepare for the next row.
        
        Example:
        ```mlir
        dsa.next_row %builder : !dsa.table_builder
        ```
    }];

    let arguments = (ins DSA_TableBuilder:$builder);
    let results = (outs);
    let assemblyFormat = "$builder attr-dict `:` type($builder)";
}

def FinalizeOp : DSA_Op<"finalize", [Pure]> {
    let summary = "convert builder to table";
    let description = [{
        Convert the result builder to a final materialized table.
        
        Example:
        ```mlir
        %table = dsa.finalize %builder : (!dsa.table_builder) -> !dsa.table
        ```
    }];

    let arguments = (ins DSA_TableBuilder:$builder);
    let results = (outs DSA_Table:$result);
    let assemblyFormat = "$builder attr-dict `:` `(` type($builder) `)` `->` type($result)";
}

#endif // PGX_DIALECT_DSA_IR_OPS