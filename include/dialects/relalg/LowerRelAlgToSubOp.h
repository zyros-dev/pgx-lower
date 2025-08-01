//===- LowerRelAlgToSubOp.h - RelAlg to SubOp lowering --------*- C++ -*-===//
//
// Lowering pass from RelAlg dialect to SubOperator dialect
// Based on LingoDB's implementation
//
//===----------------------------------------------------------------------===//

#ifndef PGX_LOWER_RELALG_TO_SUBOP_H
#define PGX_LOWER_RELALG_TO_SUBOP_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace pgx_lower::compiler::dialect::relalg {

// Create the lowering pass
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerRelAlgToSubOpPass();

// Populate conversion patterns for use in other passes
void populateRelAlgToSubOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                             mlir::TypeConverter &typeConverter);

} // namespace pgx_lower::compiler::dialect::relalg

#endif // PGX_LOWER_RELALG_TO_SUBOP_H