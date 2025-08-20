#include "execution/mlir_runner.h"
#include "execution/error_handling.h"
#include "execution/logging.h"

// Signal handling and C library includes for comprehensive exception handling
#include <csignal>
#include <cstdlib>
#include <execinfo.h> // For backtrace functions
#include <cxxabi.h> // For name demangling

// Phase-specific includes
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"

// Pipeline creation includes (moderate template weight)
#include "mlir/Passes.h"
#include "mlir/Transforms/Passes.h"

// Dialect includes needed for type checking
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"

// PostgreSQL includes for memory management
#ifdef POSTGRESQL_EXTENSION
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

// Forward declarations from main runner
extern void dumpModuleWithStats(::mlir::ModuleOp module, const std::string& title);
extern bool validateModuleState(::mlir::ModuleOp module, const std::string& phase);

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

    // Wrap PassManager run in PostgreSQL exception handling to prevent memory corruption
    bool pmRunSucceeded = false;
#ifndef BUILDING_UNIT_TESTS
    PG_TRY();
    {
#endif
        if (mlir::failed(pm.run(module))) {
            PGX_ERROR("Phase 3a failed: RelAlg→DB lowering error");
            pmRunSucceeded = false;
        }
        else {
            PGX_INFO("Phase 3a (phases): RelAlg→DB PassManager run SUCCEEDED");
            pmRunSucceeded = true;
        }
#ifndef BUILDING_UNIT_TESTS
    }
    PG_CATCH();
    {
        PGX_ERROR("Phase 3a (phases): PostgreSQL exception caught during RelAlg→DB PassManager run");
        pmRunSucceeded = false;
        // Re-throw to let PostgreSQL handle the cleanup
        PG_RE_THROW();
    }
    PG_END_TRY();
#endif

    if (!pmRunSucceeded) {
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

bool runPhase3b(::mlir::ModuleOp module) {
    auto& context = *module.getContext();

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

        ::mlir::PassManager pm(&context);
        pm.enableVerifier(true);

        mlir::pgx_lower::createDBToStandardPipeline(pm, false);
        pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DB→Standard lowering"));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after db->standard canon pass"));
        mlir::pgx_lower::createDSAToStandardPipeline(pm, false);
        pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after DSA→Standard lowering"));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::pgx_lower::createModuleDumpPass("MLIR after dsa->standard canon pass"));

        // Multi-layer exception protection around PassManager run
        bool pmRunSucceeded = false;

        // Install signal handlers to catch crashes during PassManager setup
        struct sigaction oldSigsegv, oldSigabrt;
        auto crashHandler = [](int sig) {
            PGX_ERROR("CRITICAL: Signal " + std::to_string(sig) + " caught during PassManager run");
            if (sig == SIGSEGV)
                PGX_ERROR("Segmentation fault during MLIR PassManager initialization");
            if (sig == SIGABRT)
                PGX_ERROR("Abort signal during MLIR PassManager initialization");

            // Capture stack trace
            void* stackTrace[50];
            size_t stackSize = backtrace(stackTrace, 50);
            char** stackStrings = backtrace_symbols(stackTrace, stackSize);

            PGX_ERROR("=== STACK TRACE (depth: " + std::to_string(stackSize) + ") ===");
            for (size_t i = 0; i < stackSize; i++) {
                std::string frame = stackStrings[i];

                // Try to demangle C++ names for readability
                size_t start = frame.find('(');
                size_t end = frame.find('+');
                if (start != std::string::npos && end != std::string::npos && start < end) {
                    std::string mangled = frame.substr(start + 1, end - start - 1);
                    int status;
                    char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled) {
                        frame = frame.substr(0, start + 1) + demangled + frame.substr(end);
                        free(demangled);
                    }
                }

                PGX_ERROR("  [" + std::to_string(i) + "] " + frame);
            }
            PGX_ERROR("=== END STACK TRACE ===");

            // Log additional context
            PGX_ERROR("Context: Crash during MLIR PassManager.run() before first pass execution");
            PGX_ERROR("Last successful operation: 'About to run db+dsa->standard pass'");
            PGX_ERROR("Pipeline: createDBToStandardPipeline + createDSAToStandardPipeline");

            free(stackStrings);
            _exit(1); // Force immediate exit
        };

        struct sigaction sa;
        sa.sa_handler = crashHandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGSEGV, &sa, &oldSigsegv);
        sigaction(SIGABRT, &sa, &oldSigabrt);

#ifdef POSTGRESQL_EXTENSION
        PG_TRY();
        {
#endif
            try {
                PGX_INFO("About to run db+dsa->standard pass");

                // The critical MLIR PassManager call with comprehensive protection
                if (mlir::failed(pm.run(module))) {
                    PGX_ERROR("Phase 3b failed: Unified parallel lowering error");
                    pmRunSucceeded = false;
                }
                else {
                    PGX_INFO("Phase 3b: Unified PassManager run SUCCEEDED");
                    pmRunSucceeded = true;
                }

                PGX_INFO("Finished running db+dsa->standard");

            } catch (const std::bad_alloc& e) {
                PGX_ERROR("Phase 3b: std::bad_alloc caught during PassManager run: " + std::string(e.what()));
                PGX_ERROR("Phase 3b: This indicates memory allocation failure during MLIR setup");
                pmRunSucceeded = false;
            } catch (const std::runtime_error& e) {
                PGX_ERROR("Phase 3b: std::runtime_error caught during PassManager run: " + std::string(e.what()));
                PGX_ERROR("Phase 3b: This indicates runtime failure during MLIR PassManager initialization");
                pmRunSucceeded = false;
            } catch (const std::exception& e) {
                PGX_ERROR("Phase 3b: std::exception caught during PassManager run: " + std::string(e.what()));
                PGX_ERROR("Phase 3b: This indicates C++ exception during MLIR processing");
                pmRunSucceeded = false;
            } catch (...) {
                PGX_ERROR("Phase 3b: Unknown C++ exception caught during PassManager run");
                PGX_ERROR("Phase 3b: This indicates unknown failure during MLIR PassManager setup");
                pmRunSucceeded = false;
            }
#ifdef POSTGRESQL_EXTENSION
        }
        PG_CATCH();
        {
            PGX_ERROR("Phase 3b: PostgreSQL exception caught during unified PassManager run");
            PGX_ERROR("Phase 3b: This indicates memory corruption during DB+DSA→Standard lowering");
            pmRunSucceeded = false;
            // Re-throw to let PostgreSQL handle the cleanup
            PG_RE_THROW();
        }
        PG_END_TRY();
#endif

        // Restore original signal handlers
        sigaction(SIGSEGV, &oldSigsegv, nullptr);
        sigaction(SIGABRT, &oldSigabrt, nullptr);
        PGX_INFO("Finished running db+dsa->standard");

        if (!pmRunSucceeded) {
            return false;
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
        }
        else {
            // Verify module after lowering
            if (mlir::failed(mlir::verify(module))) {
                PGX_ERROR("Phase 3c: Module verification failed after lowering");
                success = false;
            }
            else {
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
        if (!mlir::isa<mlir::ModuleOp>(op) && op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
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