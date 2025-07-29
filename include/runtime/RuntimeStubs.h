#ifndef PGX_LOWER_RUNTIME_STUBS_H
#define PGX_LOWER_RUNTIME_STUBS_H

// Stub implementations for missing LingoDB runtime classes
// These allow compilation while we integrate PostgreSQL

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace pgx_lower::compiler::runtime {

// Stub for EntryLockWrapper
class EntryLockWrapper {
public:
    static mlir::Value lock(mlir::OpBuilder& builder, mlir::Location loc) {
        // Return dummy value
        return builder.create<mlir::arith::ConstantIntOp>(loc, 0, builder.getI64Type());
    }
    
    static void unlock(mlir::OpBuilder& builder, mlir::Location loc) {
        // No-op for now
    }
};

// Stub for ThreadLocalWrapper  
class ThreadLocalWrapper {
public:
    static mlir::Value getLocal(mlir::OpBuilder& builder, mlir::Location loc) {
        // Return dummy value
        return builder.create<mlir::arith::ConstantIntOp>(loc, 0, builder.getI64Type());
    }
};

// Stub for ExecutionContextWrapper
class ExecutionContextWrapper {
public:
    static void setResult(mlir::OpBuilder& builder, mlir::Location loc) {
        // No-op for now
    }
};

// Stub for missing functions
inline void* accessHashIndex(runtime::VarLen32 description) {
    return nullptr;
}

inline uint8_t* createSimpleState(size_t sizeOfType) {
    return new uint8_t[sizeOfType];
}

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_STUBS_H