#include "standalone_mlir_runner.h"

#include <iostream>
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "pgx-lower/execution/mlir_runner.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"

namespace pgx_test {

StandalonePipelineTester::StandalonePipelineTester() {
    setupMLIRContext();
    builder_ = std::make_unique<mlir::OpBuilder>(context_.get());
}

bool StandalonePipelineTester::setupMLIRContext() {
    context_ = std::make_unique<mlir::MLIRContext>();
    return mlir_runner::setupMLIRContextForJIT(*context_);
}

mlir::OpBuilder* StandalonePipelineTester::getBuilder() {
    return builder_.get();
}

mlir::relalg::ColumnManager& StandalonePipelineTester::getColumnManager() {
    auto* dialect = context_->getLoadedDialect<mlir::relalg::RelAlgDialect>();
    if (!dialect) {
        throw std::runtime_error("RelAlg dialect not loaded");
    }
    return dialect->getColumnManager();
}

bool StandalonePipelineTester::loadRelAlgModule(const std::string& mlirText) {
    auto moduleRef = mlir::parseSourceString<mlir::ModuleOp>(mlirText, context_.get());
    if (!moduleRef) return false;
    
    module_ = std::make_unique<mlir::ModuleOp>(moduleRef.release());
    return mlir::succeeded(mlir::verify(module_->getOperation()));
}

bool StandalonePipelineTester::runPhase3a() {
    if (!module_ || !*module_) return false;
    return mlir_runner::runPhase3a(*module_);
}

bool StandalonePipelineTester::runPhase3b() {
    if (!module_ || !*module_) return false;
    return mlir_runner::runPhase3b(*module_);
}

bool StandalonePipelineTester::runPhase3c() {
    if (!module_ || !*module_) return false;
    return mlir_runner::runPhase3c(*module_);
}

std::string StandalonePipelineTester::getCurrentMLIR() const {
    if (!module_ || !*module_) return "";
    
    std::string output;
    llvm::raw_string_ostream stream(output);
    (*module_)->print(stream);
    return output;
}

bool StandalonePipelineTester::verifyCurrentModule() const {
    if (!module_ || !*module_) return false;
    return mlir::succeeded(mlir::verify(module_->getOperation()));
}

} // namespace pgx_test