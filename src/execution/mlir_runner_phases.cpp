#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

// Phase-specific includes
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"

// Pipeline creation includes (moderate template weight)  
#include "mlir/Passes.h"
#include "pgx_lower/mlir/Passes.h"

// Dialect includes needed for type checking
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"

// PostgreSQL includes for memory management
#ifndef BUILDING_UNIT_TESTS
extern "C" {
#include "postgres.h"
#include "miscadmin.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "utils/errcodes.h"
}

class Phase3bMemoryGuard;

#endif

namespace mlir_runner {

// Phase 3a: RelAlg→DB+DSA+Util lowering
bool runPhase3a(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
    context.disableMultithreading();
    
    ::mlir::PassManager pm(&context);
    pm.enableVerifier(true);

    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Phase 3a: Module verification failed before lowering");
        return false;
    }

    mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);

    dumpModuleWithStats(module, "MLIR before RelAlg -> Mixed");
    if (mlir::failed(pm.run(module))) {
        PGX_ERROR("Phase 3a failed: RelAlg→DB lowering error");
        return false;
    }

    if (mlir::failed(mlir::verify(module))) {
        PGX_ERROR("Phase 3a: Module verification failed after lowering");
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3a output")) {
        return false;
    }
    
    return true;
}

// Phase 3b: DB+DSA→Standard lowering
bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();
#ifndef BUILDING_UNIT_TESTS
    // Memory guard implementation moved to core module to avoid duplication
    // Phase3bMemoryGuard guard{};
#endif

    try {
        auto* dbDialect = context.getLoadedDialect<mlir::db::DBDialect>();
        auto* dsaDialect = context.getLoadedDialect<mlir::dsa::DSADialect>();
        auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();
        
        if (!dbDialect || !dsaDialect || !utilDialect) {
            PGX_ERROR("Phase 3b: Required dialects not loaded");
            return false;
        }
        
        if (!validateModuleState(module, "Phase 3b input")) {
            PGX_ERROR("Phase 3b: Module validation failed before running passes");
            return false;
        }
        
        if (!module) {
            PGX_ERROR("Phase 3b: Module is null!");
            return false;
        }
        
        context.disableMultithreading();
        
        if (mlir::failed(mlir::verify(module))) {
            PGX_ERROR("Phase 3b: Module verification failed before pass execution");
            return false;
        }
        
        dumpModuleWithStats(module, "MLIR after RelAlg→DB lowering");
        
        {
            ::mlir::PassManager pm(&context);
            pm.enableVerifier(true);
            
            mlir::pgx_lower::createDBToStandardPipeline(pm, false); // Don't duplicate verifier
            pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DB→Standard lowering"));
            pm.addPass(mlir::createCanonicalizerPass());
            mlir::pgx_lower::createDSAToStandardPipeline(pm, false); // Don't duplicate verifier
            pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DSA→Standard lowering"));
            pm.addPass(mlir::createCanonicalizerPass());
            if (mlir::failed(pm.run(module))) {
                PGX_ERROR("Phase 3b failed: Unified parallel lowering error");
                return false;
            }
        }
        
        if (!validateModuleState(module, "Phase 3b output")) {
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        PGX_ERROR("Phase 3b C++ exception: " + std::string(e.what()));
        return false;
    } catch (...) {
        PGX_ERROR("Phase 3b unknown C++ exception - backend crash prevented");
        return false;
    }
}

// Phase 3c: Standard→LLVM lowering
bool runPhase3c(::mlir::ModuleOp module) {
    if (!module) {
        PGX_ERROR("Phase 3c: Module is null!");
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3c input")) {
        PGX_ERROR("Phase 3c: Invalid module state before Standard→LLVM lowering");
        return false;
    }
    
    volatile bool success = false;
#ifndef BUILDING_UNIT_TESTS
    PG_TRY();
#endif
    {
        auto* moduleContext = module.getContext();
        if (!moduleContext) {
            PGX_ERROR("Phase 3c: Module context is null!");
            success = false;
            return false;
        }
        
        ::mlir::PassManager pm(moduleContext);
        pm.enableVerifier(true);
        
        if (mlir::failed(mlir::verify(module))) {
            PGX_ERROR("Phase 3c: Module verification failed before lowering");
            success = false;
            return false;
        }
        
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        if (!module) {
            success = false;
            return false;
        }

        dumpModuleWithStats(module, "MLIR before standard -> llvm");
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3c failed: Standard→LLVM lowering error");
            success = false;
        } else {
            // Verify module after lowering
            if (mlir::failed(mlir::verify(module))) {
                PGX_ERROR("Phase 3c: Module verification failed after lowering");
                success = false;
            } else {
                success = true;
            }
        }
        dumpModuleWithStats(module, "MLIR after standard -> llvm");
    }
#ifndef BUILDING_UNIT_TESTS
    PG_CATCH();
    {
        PGX_ERROR("Phase 3c: PostgreSQL exception caught during Standard→LLVM lowering");
        PG_RE_THROW();
    }
    PG_END_TRY();
#endif
    
    if (!success) {
        return false;
    }
    
    if (!validateModuleState(module, "Phase 3c output")) {
        return false;
    }
    
    // Enhanced verification: ensure all operations are LLVM dialect
    bool hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            // Special handling for func dialect which is allowed
            if (op->getDialect()->getNamespace() != "func") {
                hasNonLLVMOps = true;
            }
        }
    });

    if (hasNonLLVMOps) {
        PGX_ERROR("Phase 3c failed: Module contains non-LLVM operations");
        return false;
    }
    
    return true;
}

// Complete lowering pipeline coordinator
bool runCompleteLoweringPipeline(::mlir::ModuleOp module) {
    if (!runPhase3a(module)) {
        PGX_ERROR("Phase 3a failed");
        return false;
    }
    
    if (!runPhase3b(module)) {
        PGX_ERROR("Phase 3b failed");
        return false;
    }
    
    if (!runPhase3c(module)) {
        PGX_ERROR("Phase 3c failed");
        return false;
    }
    
    return true;
}

} // namespace mlir_runner