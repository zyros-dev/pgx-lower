#include "runtime/helpers.h"
#include "execution/logging.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"


namespace pgx_lower {
namespace compiler {
namespace runtime {

std::vector<::mlir::Value> RuntimeCallGenerator::operator()(const std::vector<::mlir::Value>& args) {
    std::vector<::mlir::Value> results;
    
    auto ptrType = mlir::LLVM::LLVMPointerType::get(this->builder.getContext());
    auto zeroOp = this->builder.create<mlir::LLVM::ZeroOp>(this->loc, ptrType);
    results.push_back(zeroOp);
    return results;
}

} // namespace runtime
} // namespace compiler
} // namespace pgx_lower