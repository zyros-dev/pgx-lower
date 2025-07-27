//===- SubOpDialect.td - SubOperator dialect definition ---*- tablegen -*-===//
//
// Based on LingoDB's SubOperator dialect
// Simplified version for pgx-lower focusing on tuple stream operations
//
//===----------------------------------------------------------------------===//

#ifndef SUBOP_DIALECT
#define SUBOP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// SubOperator dialect definition
//===----------------------------------------------------------------------===//

def SubOp_Dialect : Dialect {
    let name = "subop";
    let summary = "SubOperator dialect for tuple stream manipulation";
    let description = [{
        The SubOperator dialect provides operations for manipulating tuple streams
        with explicit state management. This is a simplified version of LingoDB's
        SubOperator dialect tailored for pgx-lower.
    }];
    let cppNamespace = "::mlir::subop";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// SubOperator types
//===----------------------------------------------------------------------===//

class SubOp_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SubOp_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// Tuple stream type - represents a stream of tuples
def SubOp_TupleStream : SubOp_Type<"TupleStream", "tuplestream"> {
    let summary = "Tuple stream type";
    let description = [{
        Represents a stream of tuples. Each tuple contains multiple columns
        of potentially different types.
    }];
    let parameters = (ins "mlir::TupleType":$tupleType);
    let assemblyFormat = "`<` $tupleType `>`";
}

// State type - represents mutable state in query execution
def SubOp_State : SubOp_Type<"State", "state"> {
    let summary = "State type for query execution";
    let description = [{
        Represents mutable state that can be used during query execution,
        such as hash tables for joins or aggregations.
    }];
    let parameters = (ins "mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

// Table type - represents a materialized table
def SubOp_Table : SubOp_Type<"Table", "table"> {
    let summary = "Table type";
    let description = [{
        Represents a materialized table with a fixed schema.
    }];
    let parameters = (ins "mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}

//===----------------------------------------------------------------------===//
// SubOperator operation base classes
//===----------------------------------------------------------------------===//

class SubOp_Op<string mnemonic, list<Trait> traits = []> :
    Op<SubOp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// SubOperator operations
//===----------------------------------------------------------------------===//

def SubOp_ScanOp : SubOp_Op<"scan", [Pure]> {
    let summary = "Scan operation - produces a tuple stream from a table";
    let description = [{
        Scans a table and produces a stream of tuples. This is the basic
        operation for reading data from a table.
    }];
    
    let arguments = (ins SubOp_Table:$table);
    let results = (outs SubOp_TupleStream:$stream);
    
    let assemblyFormat = [{
        $table attr-dict `:` type($table) `->` type($stream)
    }];
}

def SubOp_FilterOp : SubOp_Op<"filter", [SingleBlock, Pure]> {
    let summary = "Filter operation - filters tuples based on a predicate";
    let description = [{
        Filters a tuple stream based on a predicate. The predicate is
        evaluated for each tuple, and only tuples for which the predicate
        returns true are passed through.
    }];
    
    let arguments = (ins SubOp_TupleStream:$input);
    let results = (outs SubOp_TupleStream:$output);
    let regions = (region SizedRegion<1>:$predicate);
    
    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($output) $predicate
    }];
}

def SubOp_MapOp : SubOp_Op<"map", [SingleBlock, Pure]> {
    let summary = "Map operation - transforms tuples";
    let description = [{
        Applies a transformation to each tuple in the stream. The transformation
        is defined by the region, which computes new column values.
    }];
    
    let arguments = (ins SubOp_TupleStream:$input);
    let results = (outs SubOp_TupleStream:$output);
    let regions = (region SizedRegion<1>:$mapper);
    
    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($output) $mapper
    }];
}

def SubOp_MaterializeOp : SubOp_Op<"materialize"> {
    let summary = "Materialize operation - materializes a stream into a table";
    let description = [{
        Materializes a tuple stream into a table. This operation consumes
        the entire stream and stores it in memory.
    }];
    
    let arguments = (ins SubOp_TupleStream:$stream);
    let results = (outs SubOp_Table:$table);
    
    let assemblyFormat = [{
        $stream attr-dict `:` type($stream) `->` type($table)
    }];
}

def SubOp_CreateStateOp : SubOp_Op<"create_state"> {
    let summary = "Create state operation";
    let description = [{
        Creates a new state object for query execution. This can be used
        for hash tables, aggregation state, etc.
    }];
    
    let results = (outs SubOp_State:$state);
    
    let assemblyFormat = [{
        attr-dict `:` type($state)
    }];
}

def SubOp_GenerateOp : SubOp_Op<"generate", [SingleBlock]> {
    let summary = "Generate operation - generates tuples using imperative code";
    let description = [{
        Generates tuples using imperative code. The region contains code that
        can emit tuples using the emit operation.
    }];
    
    let results = (outs SubOp_TupleStream:$stream);
    let regions = (region SizedRegion<1>:$generator);
    
    let assemblyFormat = [{
        attr-dict `:` type($stream) $generator
    }];
}

def SubOp_EmitOp : SubOp_Op<"emit", [HasParent<"GenerateOp">, Terminator]> {
    let summary = "Emit operation - emits a tuple in a generate operation";
    let description = [{
        Emits a tuple within a generate operation. This operation can only
        be used within the generator region of a generate operation.
    }];
    
    let arguments = (ins Variadic<AnyType>:$values);
    
    let assemblyFormat = [{
        `(` $values `)` attr-dict `:` type($values)
    }];
}

def SubOp_YieldOp : SubOp_Op<"yield", [Pure, Terminator, 
                                       ParentOneOf<["FilterOp", "MapOp"]>]> {
    let summary = "Yield operation - yields values from a region";
    let description = [{
        Yields values from a filter or map region. For filter operations,
        yields a boolean indicating whether to keep the tuple. For map
        operations, yields the transformed values.
    }];
    
    let arguments = (ins Variadic<AnyType>:$values);
    
    let assemblyFormat = [{
        `(` $values `)` attr-dict `:` type($values)
    }];
}

#endif // SUBOP_DIALECT