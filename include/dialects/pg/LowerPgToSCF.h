#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir { namespace pg {

/// Create a pass to lower PostgreSQL dialect operations to SCF dialect
auto createLowerPgToSCFPass() -> std::unique_ptr<OperationPass<mlir::ModuleOp>>;

/// Conversion patterns for lowering pg dialect to scf/func
void populatePgToSCFConversionPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter);

}} // namespace mlir::pg