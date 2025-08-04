#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename OpT> class OperationPass;
} // namespace mlir

namespace pgx_lower { namespace compiler { namespace dialect { namespace db {

/// Create a pass to lower Database dialect operations to LLVM dialect
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createLowerDBToLLVMPass();

}}}} // namespace pgx_lower::compiler::dialect::db