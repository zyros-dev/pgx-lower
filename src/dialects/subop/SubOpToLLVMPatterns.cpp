//===----------------------------------------------------------------------===//
//
// Direct lowering patterns from SubOp to LLVM to handle remaining operations
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "dialects/subop/SubOpDialect.h"

namespace mlir {
namespace subop {

void populateSubOpToLLVMConversionPatterns(RewritePatternSet &patterns,
                                           LLVMTypeConverter &typeConverter) {
    // For now, we don't have any direct SubOp → LLVM patterns
    // All SubOp operations should be lowered through DB dialect
}

} // namespace subop
} // namespace mlir