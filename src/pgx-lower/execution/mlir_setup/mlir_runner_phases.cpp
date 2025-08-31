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

extern void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title, pgx_lower::log::Category phase);
extern bool validateModuleState(::mlir::ModuleOp module, const std::string& phase);

bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    context.disableMultithreading();

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3a: Module verification failed before lowering");
    }

    dumpModuleWithStats(module, "Phase 3a BEFORE: RelAlg -> DB+DSA+Util", pgx_lower::log::Category::RELALG_LOWER);

    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);
    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);

    // Run PassManager with pure C++ exception handling
    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3a failed: RelAlg → DB+DSA+Util lowering error");
    }

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3a: Module verification failed after lowering");
    }

    if (!validateModuleState(module, "Phase 3a output")) {
        throw std::runtime_error("Phase 3a: Module validation failed");
    }

    dumpModuleWithStats(module, "Phase 3a AFTER: RelAlg -> DB+DSA+Util", pgx_lower::log::Category::RELALG_LOWER);

    return true;
}

bool runPhase3b(::mlir::ModuleOp module) {
    PGX_LOG(DB_LOWER, TRACE, "runPhase3b: ENTERING Phase 3b DB+DSA->Standard lowering");
    auto& context = *module.getContext();

    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Checking loaded dialects");
    auto* dbDialect = context.getLoadedDialect<mlir::db::DBDialect>();
    auto* dsaDialect = context.getLoadedDialect<mlir::dsa::DSADialect>();
    auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();

    if (!dbDialect || !dsaDialect || !utilDialect) {
        PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Missing dialects - DB:%p DSA:%p Util:%p", 
                (void*)dbDialect, (void*)dsaDialect, (void*)utilDialect);
        throw std::runtime_error("Phase 3b: Required dialects not loaded");
    }
    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: All dialects loaded - DB:%p DSA:%p Util:%p", 
            (void*)dbDialect, (void*)dsaDialect, (void*)utilDialect);

    PGX_LOG(DB_LOWER, TRACE, "runPhase3b: Validating module state");
    if (!validateModuleState(module, "Phase 3b input")) {
        throw std::runtime_error("Phase 3b: Module validation failed before running passes");
    }

    if (!module) {
        throw std::runtime_error("Phase 3b: Module is null!");
    }

    context.disableMultithreading();
    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Multithreading disabled");

    PGX_LOG(DB_LOWER, TRACE, "runPhase3b: Verifying module before pass execution");
    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3b: Module verification failed before pass execution");
    }
    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Module verification passed");

    dumpModuleWithStats(module, "Phase 3b BEFORE: DB+DSA -> Standard", pgx_lower::log::Category::DB_LOWER);

    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Creating PassManager");
    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);

    PGX_LOG(DB_LOWER, TRACE, "runPhase3b: Adding DB->Standard pipeline to PassManager");
    mlir::pgx_lower::createDBToStandardPipeline(pm, false);
    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: DB->Standard pipeline added");
    
    pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DBStandard lowering", ::pgx_lower::log::Category::DB_LOWER));
    
    PGX_LOG(DB_LOWER, TRACE, "runPhase3b: Adding DSA->Standard pipeline to PassManager");
    mlir::pgx_lower::createDSAToStandardPipeline(pm, false);
    PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: DSA->Standard pipeline added");
    
    pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DSAStandard lowering", ::pgx_lower::log::Category::DSA_LOWER));

    // Run PassManager with pure C++ exception handling
    PGX_LOG(DB_LOWER, TRACE, "runPhase3b: About to run PassManager with DB+DSA lowering passes");
    try {
        PGX_LOG(DB_LOWER, TRACE, "runPhase3b: Calling pm.run(module)...");
        auto result = pm.run(module);
        PGX_LOG(DB_LOWER, TRACE, "runPhase3b: pm.run() returned, checking result");
        
        if (mlir::failed(result)) {
            PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: PassManager.run() returned FAILED status!");
            
            // Try to dump the module state after failure
            try {
                std::string moduleStr;
                llvm::raw_string_ostream stream(moduleStr);
                module.print(stream);
                PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Module state after failure (first 500 chars): %.500s", moduleStr.c_str());
            } catch (...) {
                PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: Could not dump module after failure");
            }
            
            throw std::runtime_error("Phase 3b failed: DB+DSA+Util → Standard lowering error");
        }
        PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: PassManager.run() SUCCEEDED");
    } catch (const std::exception& e) {
        PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: PassManager.run() threw exception: %s", e.what());
        throw;
    } catch (...) {
        PGX_LOG(DB_LOWER, DEBUG, "runPhase3b: PassManager.run() threw unknown exception");
        throw;
    }

    if (!validateModuleState(module, "Phase 3b output")) {
        throw std::runtime_error("Phase 3b: Module validation failed after lowering");
    }

    dumpModuleWithStats(module, "Phase 3b AFTER: DB+DSA -> Standard", pgx_lower::log::Category::DB_LOWER);

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
    pm.enableVerifier(true);

    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3c: Module verification failed before lowering");
    }

    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);

    dumpModuleWithStats(module, "Phase 3c BEFORE: Standard -> LLVM", pgx_lower::log::Category::JIT);

    if (mlir::failed(pm.run(module))) {
        throw std::runtime_error("Phase 3c failed: Standard → LLVM lowering error");
    }

    // Verify module after lowering
    if (mlir::failed(mlir::verify(module))) {
        throw std::runtime_error("Phase 3c: Module verification failed after lowering");
    }

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