#ifndef PGX_DIALECT_RELALG_IR_INTERFACES
#define PGX_DIALECT_RELALG_IR_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RelAlg Interfaces
//===----------------------------------------------------------------------===//

def RelAlgOperatorInterface : OpInterface<"RelAlgOperator"> {
    let description = [{
        Interface for relational algebra operators.
    }];
    
    let cppNamespace = "::pgx::mlir::relalg";
}

#endif // PGX_DIALECT_RELALG_IR_INTERFACES