#ifndef COMPILER_RUNTIME_HASHTABLEITERATOR_H
#define COMPILER_RUNTIME_HASHTABLEITERATOR_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace pgx_lower::compiler::runtime {

struct HashtableIterator {
    // Check if iterator has next element
    static auto hasNext(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime hasNext
            // For now, return false
            return builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI1Type(), builder.getBoolAttr(false));
        };
    }
    
    // Get next element
    static auto next(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::ValueRange {
            // TODO Phase 5: Generate actual MLIR calls to runtime next
            return {};
        };
    }
    
    // Get key at current position
    static auto getKey(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime getKey
            return args.empty() ? mlir::Value() : args[0];
        };
    }
    
    // Get value at current position
    static auto getValue(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime getValue
            return args.empty() ? mlir::Value() : args[0];
        };
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_HASHTABLEITERATOR_H