#pragma once

#include <memory>
#include <string>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

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

private:
    std::unique_ptr<mlir::MLIRContext> context_;
    std::unique_ptr<mlir::ModuleOp> module_;
    
    bool setupMLIRContext();
};

} // namespace pgx_test