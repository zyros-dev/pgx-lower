#include "standalone_mlir_runner.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

// MLIR Core Infrastructure
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

// Translation to LLVM IR
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

// Use existing mlir_runner functions
#include "pgx-lower/execution/mlir_runner.h"

namespace pgx_test {

StandalonePipelineTester::StandalonePipelineTester() 
    : phase3a_complete_(false)
    , phase3b_complete_(false) 
    , phase3c_complete_(false) {
    
    if (!setupMLIRContext()) {
        setError("Failed to setup MLIR context");
    }
}

StandalonePipelineTester::~StandalonePipelineTester() = default;

bool StandalonePipelineTester::setupMLIRContext() {
    context_ = std::make_unique<mlir::MLIRContext>();
    return mlir_runner::setupMLIRContextForJIT(*context_);
}

bool StandalonePipelineTester::loadRelAlgModule(const std::string& mlirText) {
    if (!context_) {
        setError("MLIR context not initialized");
        return false;
    }
    
    reset(); // Clear previous state
    
    auto moduleRef = mlir::parseSourceString<mlir::ModuleOp>(mlirText, context_.get());
    if (!moduleRef) {
        setError("Failed to parse MLIR text");
        return false;
    }
    
    module_ = std::make_unique<mlir::ModuleOp>(moduleRef.release());
    
    if (mlir::failed(mlir::verify(module_->getOperation()))) {
        setError("Module verification failed after loading");
        return false;
    }
    
    return true;
}

bool StandalonePipelineTester::runPhase3a() {
    if (!module_ || !*module_) {
        setError("No module loaded");
        return false;
    }
    
    if (phase3a_complete_) {
        setError("Phase 3a already completed");
        return false;
    }
    
    try {
        // Use existing mlir_runner function - pass ModuleOp directly
        bool success = mlir_runner::runPhase3a(*module_);
        if (success) {
            phase3a_complete_ = true;
            std::cout << "Phase 3a: RelAlg → DB+DSA+Util lowering completed successfully\n";
        } else {
            setError("Phase 3a failed in mlir_runner::runPhase3a");
        }
        return success;
    } catch (const std::exception& e) {
        setError("Phase 3a exception: " + std::string(e.what()));
        return false;
    }
}

bool StandalonePipelineTester::runPhase3b() {
    if (!module_ || !*module_) {
        setError("No module loaded");
        return false;
    }
    
    if (!phase3a_complete_) {
        setError("Phase 3a must be completed before Phase 3b");
        return false;
    }
    
    if (phase3b_complete_) {
        setError("Phase 3b already completed");
        return false;
    }
    
    try {
        // Use existing mlir_runner function - pass ModuleOp directly
        bool success = mlir_runner::runPhase3b(*module_);
        if (success) {
            phase3b_complete_ = true;
            std::cout << "Phase 3b: DB+DSA+Util → Standard lowering completed successfully\n";
        } else {
            setError("Phase 3b failed in mlir_runner::runPhase3b");
        }
        return success;
    } catch (const std::exception& e) {
        setError("Phase 3b exception: " + std::string(e.what()));
        return false;
    }
}

bool StandalonePipelineTester::runPhase3c() {
    if (!module_ || !*module_) {
        setError("No module loaded");
        return false;
    }
    
    if (!phase3b_complete_) {
        setError("Phase 3b must be completed before Phase 3c");
        return false;
    }
    
    if (phase3c_complete_) {
        setError("Phase 3c already completed");
        return false;
    }
    
    try {
        // Use existing mlir_runner function - pass ModuleOp directly
        bool success = mlir_runner::runPhase3c(*module_);
        if (success) {
            phase3c_complete_ = true;
            std::cout << "Phase 3c: Standard → LLVM lowering completed successfully\n";
        } else {
            setError("Phase 3c failed in mlir_runner::runPhase3c");
        }
        return success;
    } catch (const std::exception& e) {
        setError("Phase 3c exception: " + std::string(e.what()));
        return false;
    }
}

bool StandalonePipelineTester::runCompletePipeline() {
    return runPhase3a() && runPhase3b() && runPhase3c();
}

std::string StandalonePipelineTester::getCurrentMLIR() const {
    if (!module_ || !*module_) {
        return "No module loaded";
    }
    
    std::string output;
    llvm::raw_string_ostream stream(output);
    (*module_)->print(stream);
    return output;
}

std::string StandalonePipelineTester::getLLVMIR() const {
    if (!module_ || !*module_ || !phase3c_complete_) {
        return "LLVM lowering not complete";
    }
    
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module_->getOperation(), llvmContext);
    if (!llvmModule) {
        return "Failed to translate to LLVM IR";
    }
    
    std::string output;
    llvm::raw_string_ostream stream(output);
    llvmModule->print(stream, nullptr);
    return output;
}

bool StandalonePipelineTester::verifyCurrentModule() const {
    if (!module_ || !*module_) {
        return false;
    }
    
    return mlir::succeeded(mlir::verify(module_->getOperation()));
}

bool StandalonePipelineTester::containsOperations(const std::vector<std::string>& opNames) const {
    if (!module_ || !*module_) {
        return false;
    }
    
    std::string mlirText = getCurrentMLIR();
    
    for (const auto& opName : opNames) {
        if (mlirText.find(opName) == std::string::npos) {
            return false;
        }
    }
    
    return true;
}

bool StandalonePipelineTester::hasDialect(const std::string& dialectName) const {
    if (!module_ || !*module_) {
        return false;
    }
    
    bool found = false;
    (*module_)->walk([&](mlir::Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == dialectName) {
            found = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    
    return found;
}

void StandalonePipelineTester::reset() {
    module_.reset();
    phase3a_complete_ = false;
    phase3b_complete_ = false;
    phase3c_complete_ = false;
    last_error_.clear();
}

void StandalonePipelineTester::setError(const std::string& error) {
    last_error_ = error;
    std::cerr << "StandalonePipelineTester Error: " << error << std::endl;
}

// Note: Utility functions now handled by existing mlir_runner infrastructure

} // namespace pgx_test