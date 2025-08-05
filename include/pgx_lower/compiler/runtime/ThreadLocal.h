#ifndef COMPILER_RUNTIME_THREADLOCAL_H
#define COMPILER_RUNTIME_THREADLOCAL_H

#include "pgx_lower/runtime/ThreadLocal.h"
#include "pgx_lower/runtime/RuntimeFunctions.h"

namespace pgx_lower::compiler::runtime {

// MLIR wrapper functions for ThreadLocal
struct ThreadLocal {
    // MLIR wrapper that returns a function to call
    static RuntimeFunction getLocal(mlir::OpBuilder& builder, mlir::Location loc) {
        // TODO Phase 5: Implement ThreadLocal::getLocal wrapper
        return RuntimeFunction("threadlocal_get", builder, loc);
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_THREADLOCAL_H