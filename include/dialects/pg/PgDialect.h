#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

namespace pgx_lower { namespace compiler { namespace dialect { namespace pg {

using namespace mlir;

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

class PgDialect : public ::mlir::Dialect {
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

}}}} // namespace pgx_lower::compiler::dialect::pg

// Include auto-generated type declarations first
#define GET_TYPEDEF_CLASSES
#include "PgTypes.h.inc"

// Include auto-generated operation declarations
#define GET_OP_CLASSES
#include "PgDataAccess.h.inc"

// Include polymorphic operations
#define GET_OP_CLASSES
#include "PgPolymorphic.h.inc"