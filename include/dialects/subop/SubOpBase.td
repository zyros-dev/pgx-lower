//===- SubOpBase.td - SubOperator base definitions --------*- tablegen -*-===//
//
// Base definitions for SubOperator dialect
// Simplified from LingoDB for pgx-lower
//
//===----------------------------------------------------------------------===//

#ifndef SUBOP_BASE
#define SUBOP_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SubOperator dialect definition.
//===----------------------------------------------------------------------===//

def SubOp_Dialect : Dialect {
    let name = "subop";
    let summary = "A dialect around sub-operators that manipulate a tuplestream, but state is managed explicitly";
    let cppNamespace = "::pgx_lower::compiler::dialect::subop";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

// Base class for SubOperator operations
class SubOp_Op<string mnemonic, list<Trait> traits = []> : Op<SubOp_Dialect, mnemonic, traits>;

#endif // SUBOP_BASE