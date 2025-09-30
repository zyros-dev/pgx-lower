#pragma once

#include <memory>
#include <string>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"

namespace pgx_test {

class StandalonePipelineTester {
public:
    StandalonePipelineTester();

    bool loadRelAlgModule(const std::string& mlirText);
    bool runPhase3a();
    bool runPhase3b();
    bool runPhase3c();

    std::string getCurrentMLIR() const;
    bool verifyCurrentModule() const;

    mlir::MLIRContext* getContext() { return context_.get(); }
    mlir::OpBuilder* getBuilder();
    mlir::ModuleOp getModule() { return module_ ? *module_ : mlir::ModuleOp(); }
    mlir::relalg::ColumnManager& getColumnManager();

private:
    std::unique_ptr<mlir::MLIRContext> context_;
    std::unique_ptr<mlir::ModuleOp> module_;
    std::unique_ptr<mlir::OpBuilder> builder_;

    bool setupMLIRContext();
};

} // namespace pgx_test