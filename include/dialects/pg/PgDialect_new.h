#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// MLIR 20.x: Add missing operators for EmptyProperties
namespace mlir {
inline bool operator==(const EmptyProperties &, const EmptyProperties &) {
    return true;
}
inline bool operator!=(const EmptyProperties &, const EmptyProperties &) {
    return false;
}
} // namespace mlir

namespace mlir {
namespace pg {

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

class PgDialect : public Dialect {
public:
    explicit PgDialect(MLIRContext *context);

    static StringRef getDialectNamespace() { return "pg"; }

    // Parse type from string representation
    Type parseType(DialectAsmParser &parser) const override;
    
    // Print type to string representation
    void printType(Type type, DialectAsmPrinter &printer) const override;
    
    void initialize();
};

//===----------------------------------------------------------------------===//
// PostgreSQL Types (auto-generated from TableGen)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PostgreSQL Operations (auto-generated from TableGen)
//===----------------------------------------------------------------------===//

} // namespace pg
} // namespace mlir

// Include auto-generated headers from TableGen
#define GET_OP_CLASSES
#include "PgOps.h.inc"
#define GET_TYPEDEF_CLASSES
#include "PgTypes.h.inc"