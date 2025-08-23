#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace pgx_test {

class StandalonePipelineTester {
public:
    StandalonePipelineTester();
    ~StandalonePipelineTester();

    bool loadRelAlgModule(const std::string& mlirText);
    bool runPhase3a();
    bool runPhase3b();
    bool runPhase3c();
    bool runCompletePipeline();
    
    std::string getCurrentMLIR() const;
    std::string getLLVMIR() const;
    
    bool verifyCurrentModule() const;
    bool containsOperations(const std::vector<std::string>& opNames) const;
    bool hasDialect(const std::string& dialectName) const;
    
    bool isPhase3aComplete() const { return phase3a_complete_; }
    bool isPhase3bComplete() const { return phase3b_complete_; }
    bool isPhase3cComplete() const { return phase3c_complete_; }
    
    std::string getLastError() const { return last_error_; }

    void reset();

private:
    std::unique_ptr<mlir::MLIRContext> context_;
    std::unique_ptr<mlir::ModuleOp> module_;
    
    bool phase3a_complete_;
    bool phase3b_complete_;
    bool phase3c_complete_;
    std::string last_error_;
    
    bool setupMLIRContext();
    void setError(const std::string& error);
};

} // namespace pgx_test