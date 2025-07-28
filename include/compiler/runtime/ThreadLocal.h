#ifndef COMPILER_RUNTIME_THREADLOCAL_H
#define COMPILER_RUNTIME_THREADLOCAL_H

#include "runtime/ThreadLocal.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include <vector>

namespace pgx_lower::compiler::runtime {

struct ThreadLocalWrapper {
    static auto getLocal(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::SmallVector<mlir::Value> {
            // TODO: Generate MLIR calls to runtime getLocal
            // For now, return the input value
            if (!args.empty()) {
                return {args[0]};
            }
            return {};
        };
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_THREADLOCAL_H