#ifndef COMPILER_RUNTIME_HASHTABLE_H
#define COMPILER_RUNTIME_HASHTABLE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "compiler/Dialect/util/UtilOps.h"

namespace pgx_lower::compiler::runtime {

struct Hashtable {
    // Create hashtable wrapper
    static auto create(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime create
            // For now, create a null pointer
            auto ptrType = pgx_lower::compiler::dialect::util::RefType::get(
                builder.getContext(), builder.getI8Type());
            return builder.create<mlir::func::ConstantOp>(
                loc, ptrType, builder.getIntegerAttr(ptrType, 0));
        };
    }
    
    // Insert into hashtable wrapper
    static auto insert(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::ValueRange {
            // TODO Phase 5: Generate actual MLIR calls to runtime insert
            return {};
        };
    }
    
    // Create iterator wrapper
    static auto createIterator(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime createIterator
            auto ptrType = pgx_lower::compiler::dialect::util::RefType::get(
                builder.getContext(), builder.getI8Type());
            return builder.create<mlir::func::ConstantOp>(
                loc, ptrType, builder.getIntegerAttr(ptrType, 0));
        };
    }
    
    // Clear hashtable wrapper
    static auto clear(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::ValueRange {
            // TODO Phase 5: Generate actual MLIR calls to runtime clear
            return {};
        };
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_HASHTABLE_H