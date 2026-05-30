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

extern void dump_module_with_stats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase);
extern void dump_llvmir(llvm::Module* module, const std::string& title, pgx_lower::log::Category phase);
extern bool validate_module_state(::mlir::ModuleOp module, const std::string& phase);

bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    context.disableMultithreading();

    if (!validate_module_state(module, "Phase 3a input")) {
        dump_module_with_stats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module validation failed before running passes");
    }
    dump_module_with_stats(module, "Phase 3a before optimization", pgx_lower::log::Category::AST_TRANSLATE);

    mlir::PassManager pm1(&context);
    pm1.enableVerifier(false);
    pm1.addPass(mlir::createInlinerPass());
    pm1.addPass(mlir::createSymbolDCEPass());
    mlir::relalg::createQueryOptPipeline(pm1 /*, &db*/);

    if (mlir::failed(pm1.run(module))) {
        dump_module_with_stats(module, "Phase 3a AFTER: RelAlg -> Optimised RelAlg", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a failed: RelAlg → DB+DSA+Util lowering error");
    }
    dump_module_with_stats(module, "Phase 3a AFTER: RelAlg -> Optimised RelAlg", pgx_lower::log::Category::RELALG_LOWER);

    if (!validate_module_state(module, "After optimization")) {
        dump_module_with_stats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module validation failed before running passes");
    }

    ::mlir::PassManager pm(&context);
#ifndef PGX_RELEASE_MODE
    pm.enableVerifier(true);
#else
    pm.enableVerifier(false);
#endif
    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);

    // Run PassManager with pure C++ exception handling
    if (mlir::failed(pm.run(module))) {
        dump_module_with_stats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a failed: RelAlg → DB+DSA+Util lowering error");
    }

    pgx_lower::log::verify_module_or_throw(module, "Phase 3a", "Module verification failed after lowering");

    if (!validate_module_state(module, "Phase 3a output")) {
        dump_module_with_stats(module, "Failed IR", pgx_lower::log::Category::RELALG_LOWER);
        throw std::runtime_error("Phase 3a: Module validation failed");
    }

    dump_module_with_stats(module, "Phase 3a AFTER: RelAlg -> DB+DSA+Util", pgx_lower::log::Category::RELALG_LOWER);

    return true;
}

bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();

    if (!validate_module_state(module, "Phase 3b input")) {
        throw std::runtime_error("Phase 3b: Module validation failed before running passes");
    }
    context.disableMultithreading();
    dump_module_with_stats(module, "Phase 3b BEFORE: DB+DSA -> Standard", pgx_lower::log::Category::DB_LOWER);

    {
        ::mlir::PassManager pm1(&context);
#ifndef PGX_RELEASE_MODE
        pm1.enableVerifier(true);
#else
        pm1.enableVerifier(false);
#endif
        mlir::pgx_lower::createDBToStandardPipeline(pm1, false);
        if (mlir::failed(pm1.run(module))) {
            dump_module_with_stats(module, "Phase 3b failed: DB+DSA+Util → Standard lowering error", pgx_lower::log::Category::DB_LOWER);
            throw std::runtime_error("Phase 3b failed: DB+DSA+Util → Standard lowering error");
        }
        if (!validate_module_state(module, "Phase 3b output")) {
            dump_module_with_stats(module, "Phase 3b: Module validation failed after lowering", pgx_lower::log::Category::DB_LOWER);
            throw std::runtime_error("Phase 3b: Module validation failed after lowering");
        }
        dump_module_with_stats(module, "After dsa standard pipeline pm1", pgx_lower::log::Category::DB_LOWER);
    }

    {
        ::mlir::PassManager pm2(&context);
#ifndef PGX_RELEASE_MODE
        pm2.enableVerifier(true);
#else
        pm2.enableVerifier(false);
#endif
        mlir::pgx_lower::createDSAToStandardPipeline(pm2, false);
        if (mlir::failed(pm2.run(module))) {
            dump_module_with_stats(module, "Phase 3b failed: DB+DSA+Util → Standard lowering error", pgx_lower::log::Category::DB_LOWER);
            throw std::runtime_error("Phase 3b failed: DB+DSA+Util → Standard lowering error");
        }
        if (!validate_module_state(module, "Phase 3b output")) {
            dump_module_with_stats(module, "Phase 3b AFTER: RelAlg -> Optimised RelAlg", pgx_lower::log::Category::RELALG_LOWER);
            throw std::runtime_error("Phase 3b: Module validation failed after lowering");
        }
        dump_module_with_stats(module, "After dsa standard pipeline pm2", pgx_lower::log::Category::DB_LOWER);
    }

    {
        mlir::PassManager pm_func(&context, mlir::func::FuncOp::getOperationName());
#ifndef PGX_RELEASE_MODE
        pm_func.enableVerifier(true);
#else
        pmFunc.enableVerifier(false);
#endif
        pm_func.addPass(mlir::createLoopInvariantCodeMotionPass());
        pm_func.addPass(mlir::createSinkOpPass());
        pm_func.addPass(mlir::createCSEPass());
    }

    dump_module_with_stats(module, "After func pipeline", pgx_lower::log::Category::DB_LOWER);

    return true;
}

bool runPhase3c(::mlir::ModuleOp module) {
    if (!module) {
        throw std::runtime_error("Phase 3c: Module is null!");
    }

    if (!validate_module_state(module, "Phase 3c input")) {
        throw std::runtime_error("Phase 3c: Invalid module state before StandardLLVM lowering");
    }

    auto* module_context = module.getContext();
    if (!module_context) {
        throw std::runtime_error("Phase 3c: Module context is null!");
    }

    ::mlir::PassManager pm(module_context);
#ifndef PGX_RELEASE_MODE
    pm.enableVerifier(true);
#else
    pm.enableVerifier(false);
#endif

    pgx_lower::log::verify_module_or_throw(module, "Phase 3c", "Module verification failed before lowering");

    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);

    dump_module_with_stats(module, "Phase 3c BEFORE: Standard -> LLVM", pgx_lower::log::Category::JIT);

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3c failed: Standard → LLVM lowering error");
    }

    pgx_lower::log::verify_module_or_throw(module, "Phase 3c", "Module verification failed after lowering");

    dump_module_with_stats(module, "Phase 3c AFTER: Standard -> LLVM", pgx_lower::log::Category::JIT);

    if (!validate_module_state(module, "Phase 3c output")) {
        throw std::runtime_error("Phase 3c: Module validation failed after lowering");
    }

    // Ensure only LLVM operations remain
    bool has_non_llvm_ops = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            if (op->getDialect()->getNamespace() != "func") {
                has_non_llvm_ops = true;
            }
        }
    });

    if (has_non_llvm_ops) {
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
