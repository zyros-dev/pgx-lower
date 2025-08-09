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
    
    // Enable type printing/parsing
    let useDefaultTypePrinterParser = 1;
    
    let extraClassDeclaration = [{
        private:
        void registerTypes();

        public:

    }];
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

// Parameterized types following LingoDB patterns
def DSA_GenericIterable : DSA_Collection<"GenericIterable", "iterable"> {
    let summary = "generic iterable";
    let parameters = (ins "Type":$elementType, "std::string":$iteratorName);
    let hasCustomAssemblyFormat = 1;
}

def DSA_RecordBatch : DSA_Collection<"RecordBatch", "record_batch"> {
    let summary = "record_batch";
    let parameters = (ins "::mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}

def DSA_Record : DSA_Collection<"Record", "record"> {
    let summary = "record";
    let parameters = (ins "::mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}



def DSA_Flag : DSA_Type<"Flag", "flag"> {
    let summary = "binary flag";
    let description = [{
        Flag which can be used to terminate a `for` loop. Flag types are lowered 
        to local variables allocated using `alloca`.

        For example, a flag is used to terminate iteration over records when reaching 
        the output limit or to indicate whether a matching join row has been found.
    }];
}

def DSA_Vector : DSA_Collection<"Vector", "vector"> {
    let summary = "vector";
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

def DSA_AggregationHashtable : DSA_Collection<"AggregationHashtable", "aggr_ht"> {
    let summary = "aggregation hashtable";
    let parameters = (ins "::mlir::TupleType":$keyType, "::mlir::TupleType":$valType);
    let assemblyFormat = "`<` $keyType `,` $valType `>`";
}

def DSA_JoinHashtable : DSA_Collection<"JoinHashtable", "join_ht"> {
    let summary = "join hashtable";
    let parameters = (ins "::mlir::TupleType":$keyType, "::mlir::TupleType":$valType);
    let assemblyFormat = "`<` $keyType `,` $valType `>`";
}

// Table types for result materialization following LingoDB patterns
def DSA_Table : DSA_Type<"Table", "table"> {
    let summary = "materialized table";
    let assemblyFormat = "";
}

def DSA_TableBuilder : DSA_Type<"TableBuilder", "table_builder"> {
    let summary = "table builder";
    let description = [{
        Stateful table builder with a fixed schema. Values to the last row can
        be added using `ds_append` and once all columns are filled, the row can
        be finalized using `next_row`. The final table can be created using
        `finalize`.
    }];
    let parameters = (ins "::mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
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
    SingleBlockImplicitTerminator<"pgx::mlir::dsa::YieldOp">,
    RecursiveMemoryEffects
]> {
    let summary = "loop operation with implicit yield terminator";
    let description = [{
        Loop operation that iterates over an iterable with implicit yield terminator.
        The body region contains the loop body that processes each element.
        Supports iteration over different collection types following LingoDB patterns.
        
        Examples:
        ```mlir
        // Outer loop over generic iterable (record batches)
        dsa.for %batch in %iterable : !dsa.generic_iterable {
            // Process %batch
            dsa.yield
        }
        
        // Inner loop over record batch (individual records)  
        dsa.for %record in %batch : !dsa.record_batch {
            // Process %record
            dsa.yield
        }
        ```
    }];

    let arguments = (ins AnyType:$iterable);
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
// Flag Operations (Missing from LingoDB reference)
//===----------------------------------------------------------------------===//

def CreateFlagOp : DSA_Op<"createflag", [Pure]> {
    let summary = "Create flag";
    let description = [{
        Create a binary flag that can be used to control loop termination.
        Flag types are lowered to local variables allocated using `alloca`.
    }];
    let results = (outs DSA_Flag:$flag);
    let assemblyFormat = "attr-dict `:` type($flag)";
}

def SetFlagOp : DSA_Op<"setflag", []> {
    let summary = "Set flag to a value";
    let description = [{
        Set the flag to a specified boolean value.
    }];
    let arguments = (ins DSA_Flag:$flag, I1:$val);
    let assemblyFormat = "$flag `,` $val attr-dict `:` type($flag)";
}

def GetFlagOp : DSA_Op<"getflag", [Pure]> {
    let summary = "Get flag value";
    let description = [{
        Get the current value of the flag.
    }];
    let arguments = (ins DSA_Flag:$flag);
    let results = (outs I1:$res);
    let assemblyFormat = "$flag attr-dict `:` type($flag) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// Table Building Operations (Essential for MaterializeOp)
//===----------------------------------------------------------------------===//

def CreateDSOp : DSA_Op<"create_ds", [Pure]> {
    let summary = "create data structure";
    let description = [{
        Create a Vector/JoinHashtable/AggregationHashTable/TableBuilder.
        For TableBuilder, pass a string attribute with the schema description.
    }];
    let arguments = (ins Optional<AnyType>:$init_val, OptionalAttr<StrAttr>:$init_attr);
    let results = (outs AnyType:$ds);
    let assemblyFormat = "($init_attr^ `->`)? ($init_val^ `:` type($init_val) `->`)? type($ds) attr-dict";
    let builders = [
        OpBuilder<(ins "::mlir::Type":$t), [{
            build($_builder, $_state, t, ::mlir::Value(), ::mlir::StringAttr());
        }]>,
        OpBuilder<(ins "::mlir::Type":$t, "::mlir::StringAttr":$attr), [{
            build($_builder, $_state, t, ::mlir::Value(), attr);
        }]>
    ];
}

def DSAppendOp : DSA_Op<"ds_append", []> {
    let summary = "append column to Vector or last table row in a TableBuilder";
    let description = [{
        Append a value to a data structure. For TableBuilder, appends to current row
        with optional validity flag for null handling.
    }];
    let arguments = (ins AnyType:$ds, AnyType:$val, Optional<I1>:$valid);
    let assemblyFormat = "$ds `:` type($ds) `,` $val `:` type($val) (`,` $valid^)? attr-dict";
    let builders = [
        OpBuilder<(ins "::mlir::Value":$v1, "::mlir::Value":$v2), [{
            build($_builder, $_state, v1, v2, ::mlir::Value());
        }]>
    ];
}

def NextRowOp : DSA_Op<"next_row", []> {
    let summary = "start next row in a table builder";
    let description = [{
        Finalize the current row in a TableBuilder and start a new one.
    }];
    let arguments = (ins DSA_TableBuilder:$builder);
    let assemblyFormat = "$builder `:` type($builder) attr-dict";
}

def FinalizeOp : DSA_Op<"finalize", []> {
    let summary = "finalize TableBuilder or JoinHashtable";  
    let description = [{
        Finalize a data structure construction. For TableBuilder, returns the
        constructed Table.
    }];
    let arguments = (ins AnyType:$ht);
    let results = (outs Optional<AnyType>:$res);
    let assemblyFormat = "$ht `:` type($ht) (`->` type($res)^)? attr-dict";
    let builders = [
        OpBuilder<(ins "::mlir::Value":$v), [{
            build($_builder, $_state, ::mlir::TypeRange(), v);
        }]>
    ];
}

//===----------------------------------------------------------------------===//
// Hashtable Operations (Missing from LingoDB reference)
//===----------------------------------------------------------------------===//

def HashtableInsertOp : DSA_Op<"ht_insert", []> {
    let summary = "insert into a JoinHashtable or AggregationHashTable";
    let description = [{
        Insert a key-value pair into a hashtable with custom hash and equality functions.
    }];
    let arguments = (ins AnyType:$ht, AnyType:$key, Optional<AnyType>:$val);
    let regions = (region AnyRegion:$hash, AnyRegion:$equal, AnyRegion:$reduce);
    let assemblyFormat = "$ht `,` $key `:` type($key) (`,` $val^ `:` type($val))? $hash $equal $reduce attr-dict `:` type($ht)";
}

def LookupOp : DSA_Op<"lookup", [Pure]> {
    let summary = "lookup in a hash table";
    let description = [{
        Lookup entries in a hashtable that match the given key.
        Returns an iterable over matching entries.
    }];
    let arguments = (ins AnyType:$collection, AnyType:$key);
    let results = (outs DSA_GenericIterable:$iterable);
    let assemblyFormat = "$collection `:` type($collection) `,` $key `:` type($key) `->` type($iterable) attr-dict";
}

#endif // PGX_DIALECT_DSA_IR_OPS