#ifndef COMPILER_RUNTIME_ENTRYLOCKWRAPPER_H
#define COMPILER_RUNTIME_ENTRYLOCKWRAPPER_H

#include "runtime/EntryLock.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace pgx_lower::compiler::runtime {

struct EntryLockWrapper {
    static auto initialize(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) {
            // TODO Phase 5: Generate MLIR calls to runtime initialize
            return mlir::ValueRange{};
        };
    }
    
    static auto lock(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) {
            // TODO Phase 5: Generate MLIR calls to runtime lock
            return mlir::ValueRange{};
        };
    }
    
    static auto unlock(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) {
            // TODO Phase 5: Generate MLIR calls to runtime unlock
            return mlir::ValueRange{};
        };
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_ENTRYLOCKWRAPPER_H