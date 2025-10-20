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
#include "lingodb/mlir/Dialect/RelAlg/Passes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Transforms/CustomPasses.h"

class Phase3bMemoryGuard;

namespace llvm { class Module; }

namespace mlir_runner {

extern void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase);
extern void dumpLLVMIR(llvm::Module* module, const std::string& title, pgx_lower::log::Category phase);
extern bool validateModuleState(::mlir::ModuleOp module, const std::string& phase);

bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    context.disableMultithreading();

    if (!validateModuleState(module, "Phase 3a input")) {
        dumpModuleWithStats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module validation failed before running passes");
    }
    dumpModuleWithStats(module, "Phase 3a before optimization", pgx_lower::log::Category::AST_TRANSLATE);

    mlir::PassManager pm1(&context);
    pm1.enableVerifier(false);
    pm1.addPass(mlir::createInlinerPass());
    pm1.addPass(mlir::createSymbolDCEPass());
    mlir::relalg::createQueryOptPipeline(pm1 /*, &db*/);

    if (mlir::failed(pm1.run(module))) {
        dumpModuleWithStats(module, "Phase 3a AFTER: RelAlg -> Optimised RelAlg", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a failed: RelAlg → DB+DSA+Util lowering error");
    }
    dumpModuleWithStats(module, "Phase 3a AFTER: RelAlg -> Optimised RelAlg", pgx_lower::log::Category::RELALG_LOWER);

    if (!validateModuleState(module, "After optimization")) {
        dumpModuleWithStats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module validation failed before running passes");
    }

    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);
    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);

    // Run PassManager with pure C++ exception handling
    if (mlir::failed(pm.run(module))) {
        dumpModuleWithStats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a failed: RelAlg → DB+DSA+Util lowering error");
    }

#ifndef PGX_RELEASE_MODE
    if (mlir::failed(mlir::verify(module))) {
        dumpModuleWithStats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module verification failed after lowering");
    }
#endif

    if (!validateModuleState(module, "Phase 3a output")) {
        dumpModuleWithStats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module validation failed");
    }

    dumpModuleWithStats(module, "Phase 3a AFTER: RelAlg -> DB+DSA+Util", pgx_lower::log::Category::RELALG_LOWER);

    return true;
}

bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();

    if (!validateModuleState(module, "Phase 3b input")) {
        throw std::runtime_error("Phase 3b: Module validation failed before running passes");
    }
    context.disableMultithreading();
    dumpModuleWithStats(module, "Phase 3b BEFORE: DB+DSA -> Standard", pgx_lower::log::Category::DB_LOWER);

    {
        ::mlir::PassManager pm1(&context);
        pm1.enableVerifier(true);
        mlir::pgx_lower::createDBToStandardPipeline(pm1, false);
        if (mlir::failed(pm1.run(module))) {
            dumpModuleWithStats(module, "Phase 3b failed: DB+DSA+Util → Standard lowering error", pgx_lower::log::Category::DB_LOWER);
            throw std::runtime_error("Phase 3b failed: DB+DSA+Util → Standard lowering error");
        }
        if (!validateModuleState(module, "Phase 3b output")) {
            dumpModuleWithStats(module, "Phase 3b: Module validation failed after lowering", pgx_lower::log::Category::DB_LOWER);
            throw std::runtime_error("Phase 3b: Module validation failed after lowering");
        }
        dumpModuleWithStats(module, "After dsa standard pipeline pm1", pgx_lower::log::Category::DB_LOWER);
    }

    {
        ::mlir::PassManager pm2(&context);
        pm2.enableVerifier(true);
        mlir::pgx_lower::createDSAToStandardPipeline(pm2, false);
        if (mlir::failed(pm2.run(module))) {
            dumpModuleWithStats(module, "Phase 3b failed: DB+DSA+Util → Standard lowering error", pgx_lower::log::Category::DB_LOWER);
            throw std::runtime_error("Phase 3b failed: DB+DSA+Util → Standard lowering error");
        }
        if (!validateModuleState(module, "Phase 3b output")) {
            dumpModuleWithStats(module, "Phase 3b AFTER: RelAlg -> Optimised RelAlg", pgx_lower::log::Category::RELALG_LOWER);
            throw std::runtime_error("Phase 3b: Module validation failed after lowering");
        }
        dumpModuleWithStats(module, "After dsa standard pipeline pm2", pgx_lower::log::Category::DB_LOWER);
    }

    {
        mlir::PassManager pmFunc(&context, mlir::func::FuncOp::getOperationName());
        pmFunc.enableVerifier(true);
        pmFunc.addPass(mlir::createLoopInvariantCodeMotionPass());
        pmFunc.addPass(mlir::createSinkOpPass());
        pmFunc.addPass(mlir::createCSEPass());
    }

    dumpModuleWithStats(module, "After func pipeline", pgx_lower::log::Category::DB_LOWER);

    return true;
}

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
#ifndef PGX_RELEASE_MODE
    pm.enableVerifier(true);
#else
    pm.enableVerifier(false);
#endif

#ifndef PGX_RELEASE_MODE
    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3c: Module verification failed before lowering");
    }
#endif

    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);

    dumpModuleWithStats(module, "Phase 3c BEFORE: Standard -> LLVM", pgx_lower::log::Category::JIT);

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3c failed: Standard → LLVM lowering error");
    }

#ifndef PGX_RELEASE_MODE
    // Verify module after lowering
    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3c: Module verification failed after lowering");
    }
#endif

    dumpModuleWithStats(module, "Phase 3c AFTER: Standard -> LLVM", pgx_lower::log::Category::JIT);

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

    PGX_LOG(JIT, DEBUG, "Phase 3c: Standard → LLVM lowering completed successfully");
    return true;
}

bool runCompleteLoweringPipeline(::mlir::ModuleOp module) {
    runPhase3a(module);

    runPhase3b(module);

    runPhase3c(module);

    PGX_LOG(JIT, DEBUG, "Complete MLIR lowering pipeline succeeded: RelAlg → DB+DSA+Util → Standard → LLVM");
    return true;
}

} // namespace mlir_runner
