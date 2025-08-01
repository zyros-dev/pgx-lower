#ifndef PGX_LOWER_RUNTIME_STUBS_H
#define PGX_LOWER_RUNTIME_STUBS_H

// Stub implementations for missing LingoDB runtime classes
// These allow compilation while we integrate PostgreSQL

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <vector>
#include <initializer_list>
#include <cassert>

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

// Runtime function wrappers that return callable objects
// Avoid namespace conflict with ThreadLocal class
struct ThreadLocalStub {
    struct GetLocalCallable {
        mlir::OpBuilder& builder;
        mlir::Location loc;
        
        GetLocalCallable(mlir::OpBuilder& b, mlir::Location l) : builder(b), loc(l) {}
        
        std::vector<mlir::Value> operator()(std::initializer_list<mlir::Value> args) {
            // For now, just return the input value unchanged
            assert(args.size() == 1);
            return {*args.begin()};
        }
    };
    
    static GetLocalCallable getLocal(mlir::OpBuilder& builder, mlir::Location loc) {
        return GetLocalCallable(builder, loc);
    }
};

struct ExecutionContextStub {
    struct AllocStateRawCallable {
        mlir::OpBuilder& builder;
        mlir::Location loc;
        
        AllocStateRawCallable(mlir::OpBuilder& b, mlir::Location l) : builder(b), loc(l) {}
        
        std::vector<mlir::Value> operator()(std::initializer_list<mlir::Value> args) {
            // For now, return a dummy pointer
            auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
            auto nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
            return {nullPtr};
        }
    };
    
    static AllocStateRawCallable allocStateRaw(mlir::OpBuilder& builder, mlir::Location loc) {
        return AllocStateRawCallable(builder, loc);
    }
};

// More runtime stubs for missing functions
struct DataSourceIterationStub {
    struct InitCallable {
        mlir::OpBuilder& builder;
        mlir::Location loc;
        
        InitCallable(mlir::OpBuilder& b, mlir::Location l) : builder(b), loc(l) {}
        
        std::vector<mlir::Value> operator()(std::initializer_list<mlir::Value> args) {
            // Return dummy pointer for DataSourceIteration
            auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
            auto nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
            return {nullPtr};
        }
    };
    
    static InitCallable init(mlir::OpBuilder& builder, mlir::Location loc) {
        return InitCallable(builder, loc);
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_STUBS_H