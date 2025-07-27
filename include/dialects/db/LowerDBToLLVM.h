#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace db {

/// Create a pass to lower Database dialect operations to LLVM dialect
std::unique_ptr<OperationPass<ModuleOp>> createLowerDBToLLVMPass();

} // namespace db
} // namespace mlir