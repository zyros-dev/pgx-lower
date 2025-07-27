//===- LowerPgToSubOp.h - PG to SubOperator lowering -------------*- C++ -*-===//
//
// Lowering pass from PG dialect to SubOperator dialect
//
//===----------------------------------------------------------------------===//

#ifndef LOWER_PG_TO_SUBOP_H
#define LOWER_PG_TO_SUBOP_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class RewritePatternSet;
class TypeConverter;

namespace pg {

/// Populate patterns for lowering PG dialect to SubOperator dialect
void populatePgToSubOpConversionPatterns(RewritePatternSet &patterns, 
                                        TypeConverter &typeConverter);

/// Create a pass for lowering PG dialect to SubOperator dialect
std::unique_ptr<OperationPass<ModuleOp>> createLowerPgToSubOpPass();

} // namespace pg
} // namespace mlir

#endif // LOWER_PG_TO_SUBOP_H