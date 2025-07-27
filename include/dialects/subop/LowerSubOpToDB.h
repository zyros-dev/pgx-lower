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
template <typename OpT> class OperationPass;
} // namespace mlir

namespace pgx_lower { namespace compiler { namespace dialect { namespace subop {

/// Populate patterns for lowering SubOperator dialect to Database dialect
void populateSubOpToDBConversionPatterns(::mlir::RewritePatternSet &patterns, 
                                        ::mlir::TypeConverter &typeConverter);

/// Create a pass for lowering SubOperator dialect to Database dialect
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createLowerSubOpToDBPass();

}}}} // namespace pgx_lower::compiler::dialect::subop

#endif // LOWER_SUBOP_TO_DB_H