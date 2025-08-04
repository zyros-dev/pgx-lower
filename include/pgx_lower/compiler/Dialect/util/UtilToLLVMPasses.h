#ifndef PGX_LOWER_UTIL_TO_LLVM_PASSES_H
#define PGX_LOWER_UTIL_TO_LLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace pgx_lower::compiler::dialect::util {
// Stub declarations for now
std::unique_ptr<mlir::Pass> createLowerUtilToLLVMPass();
} // namespace pgx_lower::compiler::dialect::util

#endif // PGX_LOWER_UTIL_TO_LLVM_PASSES_H