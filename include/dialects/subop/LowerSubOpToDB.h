//===- LowerSubOpToDB.h - SubOperator to DB lowering -------------*- C++ -*-===//
//
// Lowering pass from SubOperator dialect to Database dialect
//
//===----------------------------------------------------------------------===//

#ifndef LOWER_SUBOP_TO_DB_H
#define LOWER_SUBOP_TO_DB_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class RewritePatternSet;
class TypeConverter;

namespace subop {

/// Populate patterns for lowering SubOperator dialect to Database dialect
void populateSubOpToDBConversionPatterns(RewritePatternSet &patterns, 
                                        TypeConverter &typeConverter);

/// Create a pass for lowering SubOperator dialect to Database dialect
std::unique_ptr<OperationPass<ModuleOp>> createLowerSubOpToDBPass();

} // namespace subop
} // namespace mlir

#endif // LOWER_SUBOP_TO_DB_H