#ifndef PGX_LOWER_MLIR_PASSES_H
#define PGX_LOWER_MLIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class PassManager;

namespace pgx_lower {

void createCompleteLoweringPipeline(PassManager& pm, bool enableVerification = false);

void createRelAlgToDBPipeline(PassManager& pm, bool enableVerification = false);
void createDBToStandardPipeline(PassManager& pm, bool enableVerification = false);
void createDSAToStandardPipeline(PassManager& pm, bool enableVerification = false);
void createStandardToLLVMPipeline(PassManager& pm, bool enableVerification = false);

std::unique_ptr<Pass> createConvertToLLVMPass();
std::unique_ptr<Pass> createModuleDumpPass(const std::string& phaseName);

void createFunctionOptimizationPipeline(PassManager& pm, bool enableVerification = false);

}
}

#endif