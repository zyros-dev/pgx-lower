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

void populateUtilToLLVMConversionPatterns(::mlir::LLVMTypeConverter& typeConverter, ::mlir::RewritePatternSet& patterns);

void populateUtilTypeConversionPatterns(::mlir::TypeConverter& typeConverter, ::mlir::RewritePatternSet& patterns);

} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_UTILTOLLVM_UTILTOLLVM_H