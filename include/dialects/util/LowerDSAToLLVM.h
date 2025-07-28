#ifndef PGX_LOWER_COMPILER_CONVERSION_UTILTOLLVM_PASSES_H
#define PGX_LOWER_COMPILER_CONVERSION_UTILTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Transforms/DialectConversion.h>

namespace pgx_lower::compiler::dialect {
namespace util {
void populateUtilToLLVMConversionPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateUtilTypeConversionPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::Pass> createUtilToLLVMPass();
void registerUtilConversionPasses();
} // end namespace util
} // end namespace pgx_lower::compiler::dialect

#endif //PGX_LOWER_COMPILER_CONVERSION_UTILTOLLVM_PASSES_H