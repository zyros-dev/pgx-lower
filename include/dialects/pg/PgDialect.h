#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

namespace mlir { namespace pg {

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

class PgDialect : public Dialect {
   public:
    explicit PgDialect(MLIRContext *context);

    static auto getDialectNamespace() -> StringRef { return "pg"; }

    void initialize();

    // TableGen will generate parseType and printType methods
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
};

//===----------------------------------------------------------------------===//
// PostgreSQL Types (auto-generated from TableGen)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PostgreSQL Operations (auto-generated from TableGen)
//===----------------------------------------------------------------------===//

}} // namespace mlir::pg

// Include auto-generated type declarations first
#define GET_TYPEDEF_CLASSES
#include "PgTypes.h.inc"

// Include auto-generated operation declarations
#define GET_OP_CLASSES
#include "PgOps.h.inc"