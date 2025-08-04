//===- SubOpBase.td - SubOperator base definitions --------*- tablegen -*-===//
//
// Base definitions for SubOperator dialect
// Simplified from LingoDB for pgx-lower
//
//===----------------------------------------------------------------------===//

#ifndef SUBOP_BASE
#define SUBOP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// SubOperator dialect definition.
//===----------------------------------------------------------------------===//

def SubOperator_Dialect : Dialect {
    let name = "subop";
    let summary = "A dialect around sub-operators that manipulate a tuplestream, but state is managed explicitly";
    let cppNamespace = "::pgx_lower::compiler::dialect::subop";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        private:
        void registerTypes();
        void registerAttrs();
        MemberManager memberManager;
        public:
        MemberManager& getMemberManager(){
            return memberManager;
        }
    }];
}

// Base class for SubOperator operations
class SubOp_Op<string mnemonic, list<Trait> traits = []> : Op<SubOperator_Dialect, mnemonic, traits>;

// Type definitions and interfaces are in SubOpOps.td and SubOpInterfaces.td

#endif // SUBOP_BASE