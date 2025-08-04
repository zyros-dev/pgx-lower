#ifndef COMPILER_RUNTIME_DATASOURCE_H
#define COMPILER_RUNTIME_DATASOURCE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "compiler/Dialect/util/UtilOps.h"

namespace pgx_lower::compiler::runtime {

struct DataSource {
    // Get data source wrapper
    static auto get(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime data source get
            // This should return a handle to the PostgreSQL table
            // For now, return the input table handle
            return args.empty() ? mlir::Value() : args[0];
        };
    }
};

struct DataSourceIteration {
    // Initialize iteration wrapper
    static auto init(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime data source init
            // This should initialize PostgreSQL table scan
            // For now, create a null pointer
            auto ptrType = pgx_lower::compiler::dialect::util::RefType::get(
                builder.getContext(), builder.getI8Type());
            return builder.create<mlir::func::ConstantOp>(
                loc, ptrType, builder.getIntegerAttr(ptrType, 0));
        };
    }
    
    // Iterate wrapper
    static auto iterate(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::ValueRange {
            // TODO Phase 5: Generate actual MLIR calls to runtime iterate
            // This should fetch next tuple batch from PostgreSQL
            // For now, return empty results
            return {};
        };
    }
    
    // Check if done wrapper
    static auto isDone(mlir::OpBuilder& builder, mlir::Location loc) {
        return [&builder, loc](mlir::ValueRange args) -> mlir::Value {
            // TODO Phase 5: Generate actual MLIR calls to runtime isDone
            // For now, return true (no rows)
            return builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI1Type(), builder.getBoolAttr(true));
        };
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_DATASOURCE_H