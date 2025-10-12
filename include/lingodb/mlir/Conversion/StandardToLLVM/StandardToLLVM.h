#ifndef MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H
#define MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace pgx_lower {

std::unique_ptr<Pass> createStandardToLLVMPass();

} // namespace pgx_lower
} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H