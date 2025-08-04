//===- DSADialect.td - Data Structure Access dialect -----*- tablegen -*-===//
//
// Based on LingoDB's DSA dialect
// Simplified version for pgx-lower focusing on data structure access
//
//===----------------------------------------------------------------------===//

#ifndef DSA_DIALECT
#define DSA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

//===----------------------------------------------------------------------===//
// Data Structure Access dialect definition
//===----------------------------------------------------------------------===//

def DSA_Dialect : Dialect {
    let name = "dsa";
    let summary = "Data Structure Access dialect";
    let description = [{
        The DSA dialect provides operations for accessing and manipulating
        data structures in a database query execution context. This is a
        simplified version of LingoDB's DSA dialect tailored for pgx-lower.
    }];
    let cppNamespace = "::pgx_lower::compiler::dialect::dsa";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Data Structure Access types
//===----------------------------------------------------------------------===//

class DSA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<DSA_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// Column type - represents a column of data
def DSA_Column : DSA_Type<"Column", "column"> {
    let summary = "Column type";
    let description = [{
        Represents a column of data with a specific element type.
        This is used for columnar data access patterns.
    }];
    let parameters = (ins "mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

// Record type - represents a row/record in a table
def DSA_Record : DSA_Type<"Record", "record"> {
    let summary = "Record type";
    let description = [{
        Represents a record (row) with a tuple type schema.
    }];
    let parameters = (ins "mlir::TupleType":$schema);
    let assemblyFormat = "`<` $schema `>`";
}

// RecordBatch type - represents a batch of records
def DSA_RecordBatch : DSA_Type<"RecordBatch", "record_batch"> {
    let summary = "Record batch type";
    let description = [{
        Represents a batch of records for vectorized processing.
    }];
    let parameters = (ins "mlir::TupleType":$schema);
    let assemblyFormat = "`<` $schema `>`";
}

// Buffer type - represents a growable buffer
def DSA_Buffer : DSA_Type<"Buffer", "buffer"> {
    let summary = "Buffer type";
    let description = [{
        Represents a growable buffer for building data structures.
    }];
    let parameters = (ins "mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

// Array type - represents a fixed-size array
def DSA_Array : DSA_Type<"Array", "array"> {
    let summary = "Array type";
    let description = [{
        Represents a fixed-size array data structure.
    }];
    let parameters = (ins "mlir::Type":$elementType, "int64_t":$size);
    let assemblyFormat = "`<` $elementType `,` $size `>`";
}

//===----------------------------------------------------------------------===//
// Data Structure Access operation base classes
//===----------------------------------------------------------------------===//

class DSA_Op<string mnemonic, list<Trait> traits = []> :
    Op<DSA_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Data Structure Access operations
//===----------------------------------------------------------------------===//

// Get record from batch
def DSA_GetRecordOp : DSA_Op<"get_record", [Pure]> {
    let summary = "Get record from batch at index";
    let description = [{
        Retrieves a record from a record batch at the specified index.
    }];
    
    let arguments = (ins DSA_RecordBatch:$batch, Index:$index);
    let results = (outs DSA_Record:$record);
    
    let assemblyFormat = [{
        $batch `[` $index `]` attr-dict `:` type($batch) `->` type($record)
    }];
}

// Access column value from record
def DSA_AtOp : DSA_Op<"at", [Pure]> {
    let summary = "Access column value from record";
    let description = [{
        Accesses a column value from a record at the specified position.
        Returns both the value and a validity flag (for NULL handling).
    }];
    
    let arguments = (ins DSA_Record:$record, Index:$position);
    let results = (outs AnyType:$value, I1:$valid);
    
    let assemblyFormat = [{
        $record `[` $position `]` attr-dict `:` type($record) `->` type($value)
    }];
}

// Get batch length
def DSA_GetBatchLengthOp : DSA_Op<"get_batch_length", [Pure]> {
    let summary = "Get number of records in batch";
    let description = [{
        Returns the number of records in a record batch.
    }];
    
    let arguments = (ins DSA_RecordBatch:$batch);
    let results = (outs Index:$length);
    
    let assemblyFormat = [{
        $batch attr-dict `:` type($batch)
    }];
}

// Create buffer
def DSA_CreateBufferOp : DSA_Op<"create_buffer"> {
    let summary = "Create a new buffer";
    let description = [{
        Creates a new growable buffer for the specified element type.
    }];
    
    let results = (outs DSA_Buffer:$buffer);
    
    let assemblyFormat = [{
        attr-dict `:` type($buffer)
    }];
}

// Append to buffer
def DSA_AppendOp : DSA_Op<"append"> {
    let summary = "Append value to buffer";
    let description = [{
        Appends a value to a growable buffer.
    }];
    
    let arguments = (ins DSA_Buffer:$buffer, AnyType:$value);
    
    let assemblyFormat = [{
        $value `to` $buffer attr-dict `:` type($value) `,` type($buffer)
    }];
}

// Create array
def DSA_CreateArrayOp : DSA_Op<"create_array"> {
    let summary = "Create a fixed-size array";
    let description = [{
        Creates a new fixed-size array.
    }];
    
    let results = (outs DSA_Array:$array);
    
    let assemblyFormat = [{
        attr-dict `:` type($array)
    }];
}

// Array access
def DSA_ArrayGetOp : DSA_Op<"array_get", [Pure]> {
    let summary = "Get element from array";
    let description = [{
        Retrieves an element from an array at the specified index.
    }];
    
    let arguments = (ins DSA_Array:$array, Index:$index);
    let results = (outs AnyType:$value);
    
    let assemblyFormat = [{
        $array `[` $index `]` attr-dict `:` type($array) `->` type($value)
    }];
}

// Array set
def DSA_ArraySetOp : DSA_Op<"array_set"> {
    let summary = "Set element in array";
    let description = [{
        Sets an element in an array at the specified index.
    }];
    
    let arguments = (ins DSA_Array:$array, Index:$index, AnyType:$value);
    
    let assemblyFormat = [{
        $array `[` $index `]` `=` $value attr-dict `:` type($array) `,` type($value)
    }];
}

// Load from memory
def DSA_LoadOp : DSA_Op<"load", [MemoryEffects<[MemRead]>]> {
    let summary = "Load value from memory";
    let description = [{
        Loads a value from a memory location. This is used for low-level
        data access within data structures.
    }];
    
    let arguments = (ins LLVM_AnyPointer:$ptr);
    let results = (outs AnyType:$value);
    
    let assemblyFormat = [{
        $ptr attr-dict `:` type($ptr) `->` type($value)
    }];
}

// Store to memory
def DSA_StoreOp : DSA_Op<"store", [MemoryEffects<[MemWrite]>]> {
    let summary = "Store value to memory";
    let description = [{
        Stores a value to a memory location. This is used for low-level
        data access within data structures.
    }];
    
    let arguments = (ins AnyType:$value, LLVM_AnyPointer:$ptr);
    
    let assemblyFormat = [{
        $value `,` $ptr attr-dict `:` type($value) `,` type($ptr)
    }];
}

#endif // DSA_DIALECT