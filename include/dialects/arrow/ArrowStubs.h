#ifndef PGX_LOWER_ARROW_STUBS_H
#define PGX_LOWER_ARROW_STUBS_H

// Stub Arrow dialect to avoid compilation errors
// These will be replaced with PostgreSQL runtime calls

#include "mlir/IR/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

namespace pgx_lower::compiler::dialect::arrow {

// Stub Arrow array type
class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type, mlir::DefaultTypeStorage> {
public:
    using Base::Base;
    static ArrayType get(mlir::MLIRContext* context) {
        // Return a stub type - we'll replace this anyway
        return Base::get(context);
    }
};

// Stub BuilderFromPtr operation
struct BuilderFromPtr {
    static mlir::Value create(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value ptr) {
        // This will be replaced with PostgreSQL-specific code
        // For now, just return the pointer as-is
        return ptr;
    }
};

} // namespace pgx_lower::compiler::dialect::arrow

#endif // PGX_LOWER_ARROW_STUBS_H