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
namespace util {

/// Create the Util to LLVM conversion pass
std::unique_ptr<::mlir::Pass> createUtilToLLVMPass();

/// Populate Util to LLVM conversion patterns
void populateUtilToLLVMConversionPatterns(::mlir::LLVMTypeConverter& typeConverter, ::mlir::RewritePatternSet& patterns);

/// Populate Util type conversion patterns  
void populateUtilTypeConversionPatterns(::mlir::TypeConverter& typeConverter, ::mlir::RewritePatternSet& patterns);

} // namespace util
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_UTILTOLLVM_UTILTOLLVM_H