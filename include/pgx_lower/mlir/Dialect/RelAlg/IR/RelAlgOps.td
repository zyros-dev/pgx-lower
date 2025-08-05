#ifndef PGX_DIALECT_RELALG_IR_OPS
#define PGX_DIALECT_RELALG_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td" 
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "RelAlgInterfaces.td"

//===----------------------------------------------------------------------===//
// RelAlg dialect definition.
//===----------------------------------------------------------------------===//

def RelAlg_Dialect : Dialect {
    let name = "relalg";
    let summary = "A dialect for relational algebra in pgx-lower";
    let cppNamespace = "::pgx::mlir::relalg";
    
    // Disable properties system to avoid namespace issues
    let usePropertiesForAttributes = 0;

    let extraClassDeclaration = [{
        private:
            ColumnManager columnManager;
        public:
            ColumnManager& getColumnManager(){
                return columnManager;
            }
    }];
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class RelAlg_Type<string name, string typeMnemonic> : TypeDef<RelAlg_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def RelAlg_TupleStream : RelAlg_Type<"TupleStream", "tuplestream"> {
    let summary = "tuple stream type";
}

def RelAlg_Tuple : RelAlg_Type<"Tuple", "tuple"> {
    let summary = "tuple type";
}

def RelAlg_Table : RelAlg_Type<"Table", "table"> {
    let summary = "table type";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class RelAlg_Attr<string name> : AttrDef<RelAlg_Dialect, name>;

def ColumnRefAttr : RelAlg_Attr<"ColumnRef"> {
    let mnemonic = "columnref";
    let parameters = (ins "std::string":$name);
    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Base RelAlg operation definition.
//===----------------------------------------------------------------------===//

class RelAlg_Op<string mnemonic, list<Trait> traits = []> : Op<RelAlg_Dialect, mnemonic, traits> {
    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Operations that "create" relations
//===----------------------------------------------------------------------===//

def BaseTableOp : RelAlg_Op<"basetable",
        [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "base table operation";
    let description = [{
        Represents access to a base table in the database.
        Creates a tuple stream from the specified table.
    }];

    let arguments = (ins StrAttr:$table_name);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//

def ProjectionOp : RelAlg_Op<"projection", [Pure]> {
    let summary = "projection operation";
    let description = [{
        Projects specified columns from the input relation.
        Equivalent to SELECT clause in SQL.
    }];

    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs RelAlg_TupleStream:$result);
}

def SelectionOp : RelAlg_Op<"selection", [Pure]> {
    let summary = "selection operation";  
    let description = [{
        Filter tuple stream based on a predicate.
        The region returns 1 iff the tuple should be in the output.
        Equivalent to WHERE clause in SQL.
    }];

    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
}

def MapOp : RelAlg_Op<"map", [Pure]> {
    let summary = "map operation";
    let description = [{
        Computes additional columns for each tuple in the stream.
        The region defines the computation for new columns.
    }];

    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$computation);
}

def MaterializeOp : RelAlg_Op<"materialize", [Pure]> {
    let summary = "materialize operation";
    let description = [{
        Materializes a tuple stream into a table that can be returned.
        This is the final operation that converts internal tuple streams
        to returnable results.
    }];

    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs RelAlg_Table:$result);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Terminators
////////////////////////////////////////////////////////////////////////////////////////////////////

def ReturnOp : RelAlg_Op<"return", [Pure, ReturnLike, Terminator]> {
    let summary = "relational algebra return operation";
    let description = [{
        Terminator for relational algebra operations. Returns values from regions.
    }];

    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////

def GetColumnOp : RelAlg_Op<"getcol", [Pure]> {
    let summary = "get column operation";
    let description = [{
        Extract a column value from a tuple.
        Used within regions to access tuple data.
    }];

    let arguments = (ins StrAttr:$column_name, RelAlg_Tuple:$tuple);
    let results = (outs AnyType:$res);
}

#endif // PGX_DIALECT_RELALG_IR_OPS