//===----------------------------------------------------------------------===//
// RelAlg Dialect Base
//===----------------------------------------------------------------------===//

#ifndef RELALG_BASE
#define RELALG_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RelAlg dialect definition.
//===----------------------------------------------------------------------===//

def RelAlg_Dialect : Dialect {
    let name = "relalg";
    let summary = "A dialect for relational algebra (simplified from LingoDB)";
    let cppNamespace = "pgx_lower::compiler::dialect::relalg";
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base RelAlg operation definition.
//===----------------------------------------------------------------------===//

class RelAlg_Op<string mnemonic, list<Trait> traits = []> : 
    Op<RelAlg_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

#endif // RELALG_BASE