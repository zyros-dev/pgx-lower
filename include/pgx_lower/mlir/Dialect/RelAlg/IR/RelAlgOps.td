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
    let assemblyFormat = "";
}

def RelAlg_Tuple : RelAlg_Type<"Tuple", "tuple"> {
    let summary = "tuple type"; 
    let assemblyFormat = "";
}

def RelAlg_Table : RelAlg_Type<"Table", "table"> {
    let summary = "table type";
    let assemblyFormat = "";
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

    let arguments = (ins StrAttr:$table_name, I64Attr:$table_oid);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$table_name $table_oid $body attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//


def MaterializeOp : RelAlg_Op<"materialize", [Pure]> {
    let summary = "materialize operation";
    let description = [{
        Materializes a tuple stream into a table that can be returned.
        This is the final operation that converts internal tuple streams
        to returnable results.
    }];

    let arguments = (ins RelAlg_TupleStream:$rel, ArrayAttr:$columns);
    let results = (outs RelAlg_Table:$result);
    let assemblyFormat = "$rel $columns attr-dict `:` type($rel) `->` type($result)";
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
    let assemblyFormat = "$column_name `from` $tuple attr-dict `:` type($tuple) `->` type($res)";
}

#endif // PGX_DIALECT_RELALG_IR_OPS