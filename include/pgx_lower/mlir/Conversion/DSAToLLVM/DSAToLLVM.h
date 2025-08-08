#ifndef PGX_LOWER_MLIR_CONVERSION_DSATOLLVM_DSATOLLVM_H
#define PGX_LOWER_MLIR_CONVERSION_DSATOLLVM_DSATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pgx_conversion {

//===----------------------------------------------------------------------===//
// DSA to LLVM Type Converter
//===----------------------------------------------------------------------===//

class DSAToLLVMTypeConverter : public LLVMTypeConverter {
public:
    explicit DSAToLLVMTypeConverter(MLIRContext *ctx);
};

//===----------------------------------------------------------------------===//
// DSA to LLVM Conversion Patterns
//===----------------------------------------------------------------------===//

// Forward declarations for conversion patterns
class CreateDSToLLVMPattern;
class ScanSourceToLLVMPattern;
class FinalizeToLLVMPattern;
class DSAppendToLLVMPattern;
class NextRowToLLVMPattern;
class ForOpToLLVMPattern;
class YieldOpToLLVMPattern;

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

/// Create a pass that lowers DSA operations to LLVM IR
std::unique_ptr<Pass> createDSAToLLVMPass();

/// Register the DSA to LLVM conversion passes
void registerDSAToLLVMConversionPasses();

} // namespace pgx_conversion
} // namespace mlir

#endif // PGX_LOWER_MLIR_CONVERSION_DSATOLLVM_DSATOLLVM_H