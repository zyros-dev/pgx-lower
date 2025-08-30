#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "lingodb/mlir/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pgx-lower/utility/logging.h"
#include "llvm/Support/FormatVariadic.h"
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// Additional includes for unified conversion pass
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pgx_lower {

// Forward declarations
std::unique_ptr<Pass> createConvertToLLVMPass();
std::unique_ptr<Pass> createStandardToLLVMPass();


// Phase 1: RelAlg→DB lowering pipeline (using LingoDB's pipeline)
void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("createRelAlgToDBPipeline: Adding relalg to db pipeline");
    
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    relalg::createLowerRelAlgPipeline(pm);
    // Note: LingoDB's function already adds canonicalizer, no need to add again
}

// Phase 2a: DB→Standard lowering pipeline (using LingoDB's pipeline)
void createDBToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("[createDBToStandardPipeline]: Adding DB to Standard pipeline (Phase 2a)");
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    db::createLowerDBPipeline(pm);
    
    // Add canonicalizer after LingoDB's pipeline
    pm.addPass(createCanonicalizerPass());
}

// Phase 2b: DSAStandard lowering pipeline (following LingoDB sequential pattern)
void createDSAToStandardPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("[createDSAToStandardPipeline]: Adding DSA to Standard pipeline (Phase 2b)");
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    auto dsaPass = dsa::createLowerToStdPass();
    if (!dsaPass) {
        PGX_ERROR("createDSAToStandardPipeline: DSA pass creation returned null!");
        return;
    }
    pm.addPass(std::move(dsaPass));
    pm.addPass(createCanonicalizerPass());
}

void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification) {
    PGX_INFO("[createStandardToLLVMPipeline]: Adding DB DSA to Standard pipeline");
    if (enableVerification) {
        pm.enableVerifier(true);
    }
    
    pm.addPass(std::move(createStandardToLLVMPass()));
    pm.addPass(createCanonicalizerPass());
}

// Debug pass for module dumping between lowering stages
class ModuleDumpPass : public PassWrapper<ModuleDumpPass, OperationPass<ModuleOp>> {
private:
    std::string phaseName;
    
public:
    ModuleDumpPass(const std::string& name) : phaseName(name) {}
    
    void runOnOperation() override {
        auto module = getOperation();
        
        if (!module) {
            PGX_WARNING("ModuleDumpPass: Module is null for phase: " + phaseName);
            return;
        }

        try {
            auto timestamp = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(timestamp);
            auto tm = *std::localtime(&time_t);
            std::ostringstream timestampStr;
            timestampStr << std::put_time(&tm, "%Y%m%d_%H%M%S");
            
            PGX_INFO(" ");
            PGX_INFO("=== MLIR MODULE CONTENT: " + phaseName + " ===");
            
            // Convert module to string and log line by line
            std::string moduleStr;
            llvm::raw_string_ostream stream(moduleStr);
            module.print(stream);
            
            std::istringstream iss(moduleStr);
            std::string line;
            int lineNum = 1;
            while (std::getline(iss, line)) {
                std::string formattedLine = llvm::formatv("{0,3}: {1}", lineNum++, line);
                PGX_INFO(formattedLine);
            }
            
            PGX_INFO("=== END MLIR MODULE CONTENT ===");
            
            // Optional: Dump to file for external debugging  
            std::string filename = "/tmp/pgx_lower_" + phaseName + "_" + timestampStr.str() + ".mlir";
            std::replace(filename.begin(), filename.end(), ' ', '_');
            std::replace(filename.begin(), filename.end(), '-', '_');
            
            std::ofstream file(filename);
            if (file.is_open()) {
                file << moduleStr;
                file.close();
                PGX_INFO("Module dumped to: " + filename);
            }
            
            PGX_INFO("=== End Module Debug Dump ===");
            PGX_INFO(" ");
            
        } catch (const std::exception& e) {
            PGX_ERROR("Exception in ModuleDumpPass: " + std::string(e.what()));
        } catch (...) {
            PGX_ERROR("Unknown exception in ModuleDumpPass");
        }
    }
    
    llvm::StringRef getArgument() const override { return "module-dump"; }
    llvm::StringRef getDescription() const override { 
        return "Dump MLIR module for debugging"; 
    }
};

std::unique_ptr<Pass> createModuleDumpPass(const std::string& phaseName) {
    return std::make_unique<ModuleDumpPass>(phaseName);
}

} // namespace pgx_lower
} // namespace mlir