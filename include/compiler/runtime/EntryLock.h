#ifndef COMPILER_RUNTIME_ENTRYLOCK_H
#define COMPILER_RUNTIME_ENTRYLOCK_H

#include "runtime/EntryLock.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace pgx_lower::compiler::runtime {

struct EntryLock {
    static auto unlock(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](std::initializer_list<mlir::Value> args) {
            // TODO: Generate MLIR calls to runtime unlock
        };
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_ENTRYLOCK_H