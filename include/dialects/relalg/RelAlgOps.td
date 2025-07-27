//===----------------------------------------------------------------------===//
// RelAlg Operations (Simplified from LingoDB)
//===----------------------------------------------------------------------===//

#ifndef RELALG_OPS
#define RELALG_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "dialects/relalg/RelAlgBase.td"
include "dialects/relalg/RelAlgTypes.td"

//===----------------------------------------------------------------------===//
// Operations that "create" relations
//===----------------------------------------------------------------------===//

def RelAlg_BaseTableOp : RelAlg_Op<"basetable"> {
    let summary = "Base table operation - represents scanning a PostgreSQL table";
    let description = [{
        The basetable operation represents scanning a base table in the database.
        This is equivalent to a PostgreSQL SeqScan operation.
    }];

    let arguments = (ins StrAttr:$table_identifier);
    let results = (outs TupleStream:$result);
    
    let assemblyFormat = [{
        `{` `table_identifier` `=` $table_identifier `}` attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//

def RelAlg_SelectionOp : RelAlg_Op<"selection"> {
    let summary = "Selection operation - represents WHERE clause filtering";
    let description = [{
        Filter tuple stream based on a predicate. The region contains the 
        predicate logic that operates on a single tuple and returns a boolean.
    }];

    let arguments = (ins TupleStream:$rel);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    
    let assemblyFormat = [{
        $rel $predicate attr-dict `:` type($rel) `->` type($result)
    }];
}

def RelAlg_ProjectionOp : RelAlg_Op<"projection"> {
    let summary = "Projection operation - represents SELECT clause";
    let description = [{
        Project specific columns from the input tuple stream. The region contains
        the computation logic for the projected columns.
    }];

    let arguments = (ins TupleStream:$rel);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$computation);
    
    let assemblyFormat = [{
        $rel $computation attr-dict `:` type($rel) `->` type($result)
    }];
}

def RelAlg_MaterializeOp : RelAlg_Op<"materialize"> {
    let summary = "Materialize operation - produces final query results";
    let description = [{
        Materializes the tuple stream into final query results that can be
        returned to PostgreSQL.
    }];

    let arguments = (ins TupleStream:$rel);
    let results = (outs);  // No MLIR result - produces PostgreSQL results
    
    let assemblyFormat = [{
        $rel attr-dict `:` type($rel)
    }];
}

#endif // RELALG_OPS