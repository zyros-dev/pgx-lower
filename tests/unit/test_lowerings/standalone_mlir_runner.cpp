#include "standalone_mlir_runner.h"

#include <iostream>
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "pgx-lower/execution/mlir_runner.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace pgx_test {

StandalonePipelineTester::StandalonePipelineTester() {
    setupMLIRContext();
    builder_ = std::make_unique<mlir::OpBuilder>(context_.get());
}

bool StandalonePipelineTester::setupMLIRContext() {
    // Test harness wants a fresh MLIRContext per test to avoid state
    // bleed between TEST_F instances. Load the same dialect set that
    // production (MLIRRuntime singleton) loads.
    context_ = std::make_unique<mlir::MLIRContext>();
    context_->disableMultithreading();
    context_->getOrLoadDialect<mlir::func::FuncDialect>();
    context_->getOrLoadDialect<mlir::arith::ArithDialect>();
    context_->getOrLoadDialect<mlir::scf::SCFDialect>();
    context_->getOrLoadDialect<mlir::memref::MemRefDialect>();
    context_->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context_->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context_->getOrLoadDialect<mlir::relalg::RelAlgDialect>();
    context_->getOrLoadDialect<mlir::db::DBDialect>();
    context_->getOrLoadDialect<mlir::dsa::DSADialect>();
    context_->getOrLoadDialect<mlir::util::UtilDialect>();
    return true;
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