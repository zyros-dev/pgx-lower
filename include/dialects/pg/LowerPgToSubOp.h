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
template <typename OpT> class OperationPass;
} // namespace mlir

namespace pgx_lower { namespace compiler { namespace dialect { namespace pg {

/// Populate patterns for lowering PG dialect to SubOperator dialect
void populatePgToSubOpConversionPatterns(::mlir::RewritePatternSet &patterns, 
                                        ::mlir::TypeConverter &typeConverter);

/// Create a pass for lowering PG dialect to SubOperator dialect
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createLowerPgToSubOpPass();

}}}} // namespace pgx_lower::compiler::dialect::pg

#endif // LOWER_PG_TO_SUBOP_H