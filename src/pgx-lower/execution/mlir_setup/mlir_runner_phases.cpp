#include "pgx-lower/execution/mlir_runner.h"
#include "pgx-lower/utility/error_handling.h"
#include "pgx-lower/utility/logging.h"

#include <stdexcept>
#include <string>

#include <csignal>
#include <cstdlib>
#include <execinfo.h>
#include <cxxabi.h>

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"

#include "lingodb/mlir/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

class Phase3bMemoryGuard;

namespace mlir_runner {

extern void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title);
extern bool validateModuleState(::mlir::ModuleOp module, const std::string& phase);

bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    context.disableMultithreading();

    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3a: Module verification failed before lowering");
    }

    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);

    dumpModuleWithStats(module, "MLIR before RelAlg -> Mixed");

    // Run PassManager with pure C++ exception handling
    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3a failed: RelAlg → DB+DSA+Util lowering error");
    }

    PGX_INFO("Phase 3a: RelAlg → DB+DSA+Util PassManager run SUCCEEDED");

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3a: Module verification failed after lowering");
    }

    if (!validateModuleState(module, "Phase 3a output")) {
        throw std::runtime_error("Phase 3a: Module validation failed");
    }

    return true;
}

bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();

    auto* dbDialect = context.getLoadedDialect<mlir::db::DBDialect>();
    auto* dsaDialect = context.getLoadedDialect<mlir::dsa::DSADialect>();
    auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();

    if (!dbDialect || !dsaDialect || !utilDialect) {
        throw std::runtime_error("Phase 3b: Required dialects not loaded");
    }

    if (!validateModuleState(module, "Phase 3b input")) {
        throw std::runtime_error("Phase 3b: Module validation failed before running passes");
    }

    if (!module) {
        throw std::runtime_error("Phase 3b: Module is null!");
    }

    context.disableMultithreading();

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3b: Module verification failed before pass execution");
    }

    dumpModuleWithStats(module, "MLIR after RelAlgDB lowering");

    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);

    mlir::pgx_lower::createDBToStandardPipeline(pm, false);
    pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DBStandard lowering"));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after db->standard canon pass"));
    mlir::pgx_lower::createDSAToStandardPipeline(pm, false);
    pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DSAStandard lowering"));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after dsa->standard canon pass"));

    PGX_INFO("About to run db+dsa->standard pass");

    // Run PassManager with pure C++ exception handling
    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3b failed: DB+DSA+Util → Standard lowering error");
    }

    PGX_INFO("Phase 3b: DB+DSA+Util → Standard PassManager run SUCCEEDED");

    if (!validateModuleState(module, "Phase 3b output")) {
        throw std::runtime_error("Phase 3b: Module validation failed after lowering");
    }

    return true;
}

// Phase 3c: StandardLLVM lowering
bool runPhase3c(::mlir::ModuleOp module) {
    if (!module) {
        throw std::runtime_error("Phase 3c: Module is null!");
    }

    if (!validateModuleState(module, "Phase 3c input")) {
        throw std::runtime_error("Phase 3c: Invalid module state before StandardLLVM lowering");
    }

    auto* moduleContext = module.getContext();
    if (!moduleContext) {
        throw std::runtime_error("Phase 3c: Module context is null!");
    }

    ::mlir::PassManager pm(moduleContext);
    pm.enableVerifier(true);

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3c: Module verification failed before lowering");
    }

    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);

    dumpModuleWithStats(module, "MLIR before standard -> llvm");

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3c failed: Standard → LLVM lowering error");
    }

    // Verify module after lowering
    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3c: Module verification failed after lowering");
    }

    dumpModuleWithStats(module, "MLIR after standard -> llvm");

    if (!validateModuleState(module, "Phase 3c output")) {
        throw std::runtime_error("Phase 3c: Module validation failed after lowering");
    }

    // Ensure only LLVM operations remain
    bool hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            if (op->getDialect()->getNamespace() != "func") {
                hasNonLLVMOps = true;
            }
        }
    });

    if (hasNonLLVMOps) {
        throw std::runtime_error("Phase 3c failed: Module contains non-LLVM operations after lowering");
    }

    PGX_INFO("Phase 3c: Standard → LLVM lowering completed successfully");
    return true;
}

bool runCompleteLoweringPipeline(::mlir::ModuleOp module) {
    runPhase3a(module);

    runPhase3b(module);

    runPhase3c(module);

    PGX_INFO("Complete MLIR lowering pipeline succeeded: RelAlg → DB+DSA+Util → Standard → LLVM");
    return true;
}

} // namespace mlir_runner