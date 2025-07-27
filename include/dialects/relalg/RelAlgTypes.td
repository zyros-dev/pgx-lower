//===----------------------------------------------------------------------===//
// RelAlg Types
//===----------------------------------------------------------------------===//

#ifndef RELALG_TYPES
#define RELALG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "dialects/relalg/RelAlgBase.td"

//===----------------------------------------------------------------------===//
// RelAlg type definitions.
//===----------------------------------------------------------------------===//

class RelAlg_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RelAlg_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def RelAlg_TupleStreamType : RelAlg_Type<"TupleStream", "tuplestream"> {
    let summary = "A stream of tuples";
    let description = [{
        Represents a stream of tuples flowing through relational operations.
        This is the primary data type for RelAlg operations.
    }];
}

// Convenience type predicate for use in operation definitions
def TupleStream : Type<CPred<"mlir::isa<::mlir::relalg::TupleStreamType>($_self)">, 
                      "tuple stream type">;

#endif // RELALG_TYPES