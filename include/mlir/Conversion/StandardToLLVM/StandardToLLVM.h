#ifndef MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H
#define MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace pgx_lower {

/// Create a pass to convert all standard dialects (SCF, Func, Arith, ControlFlow, Util)
/// to LLVM dialect in a unified manner. This ensures all conversion patterns are
/// applied together, preventing issues with missing patterns.
std::unique_ptr<Pass> createStandardToLLVMPass();

} // namespace pgx_lower
} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H