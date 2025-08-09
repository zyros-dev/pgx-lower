#ifndef PGX_LOWER_MLIR_CONVERSION_UTILTOLLVM_UTILTOLLVM_H
#define PGX_LOWER_MLIR_CONVERSION_UTILTOLLVM_UTILTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class TypeConverter;
class RewritePatternSet;
class LLVMTypeConverter;
} // namespace mlir

namespace pgx {
namespace mlir {

/// Populate Util to LLVM conversion patterns
void populateUtilToLLVMConversionPatterns(::mlir::LLVMTypeConverter &typeConverter,
                                          ::mlir::RewritePatternSet &patterns);

/// Create a pass to convert Util operations to LLVM dialect
std::unique_ptr<::mlir::Pass> createUtilToLLVMPass();

} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_UTILTOLLVM_UTILTOLLVM_H