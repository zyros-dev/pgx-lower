#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/error_handling.h"
#include "core/postgresql_ast_translator.h"
// PG dialect removed - using RelAlg instead
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpToControlFlow.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/LowerDBToDSA.h"
#include "dialects/dsa/DSADialect.h"
#include "dialects/util/UtilToLLVMPasses.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"

#include <fstream>
#include "llvm/IR/Verifier.h"

// Prevent libintl.h conflicts with PostgreSQL macros
// This is a bit strange to me - so LLVM drags in some macros from libintl.h
// and those conflict with things inside of libintl.h. So this should resolve
// those problems?
#define ENABLE_NLS 0

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/TargetSelect.h"

// Runtime symbols will be registered from the interface functions
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <sstream>
#include <cstring>
#include <signal.h>
#include <cstdio> // For fprintf
#include <atomic>

// Include runtime functions after all LLVM/MLIR headers to avoid macro conflicts
#include "runtime/tuple_access.h"

namespace mlir_runner {

// MLIR Execution Engine - handles lowering and JIT execution
bool executeMLIRModule(mlir::ModuleOp &module, MLIRLogger &logger) {
    logger.notice("About to initialize native target and create ExecutionEngine...");
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    logger.notice("Native target initialized successfully");

    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    // Register all needed dialects
    registry.insert<pgx_lower::compiler::dialect::relalg::RelAlgDialect>();
    registry.insert<pgx_lower::compiler::dialect::subop::SubOperatorDialect>();
    registry.insert<pgx_lower::compiler::dialect::db::DBDialect>();
    registry.insert<pgx_lower::compiler::dialect::dsa::DSADialect>();
    registry.insert<pgx_lower::compiler::dialect::util::UtilDialect>();
    registry.insert<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
    auto &context = *module.getContext();
    context.appendDialectRegistry(registry);

    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    // Verify the module before ExecutionEngine creation
    if (failed(mlir::verify(module))) {
        logger.error("MLIR module verification failed before ExecutionEngine creation");
        module.dump();
        return false;
    }
    logger.notice("MLIR module verification passed - proceeding to lowering");
    
    logger.notice("=== Running FULL LingoDB lowering pipeline ===");
    
    // Check for dialect operations
    bool hasRelAlgOps = false;
    module.walk([&](mlir::Operation* op) {
        if (op->getName().getDialectNamespace() == "relalg") {
            hasRelAlgOps = true;
            logger.debug("Found RelAlg operation: " + op->getName().getStringRef().str());
        }
    });
    
    if (!hasRelAlgOps) {
        logger.error("No RelAlg operations found in module - AST translation may have failed");
        module.dump();
        return false;
    }
    
    auto pm = mlir::PassManager(&context);
    
    // Skip initial dump to reduce output
    logger.notice("Starting LingoDB lowering pipeline...");
    
    // Run each phase separately to isolate crashes
    
    // Phase 1: RelAlg → SubOp lowering
    logger.notice("Phase 1: RelAlg → SubOp lowering");
    {
        auto pm1 = mlir::PassManager(&context);
        pm1.addPass(pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass());
        if (failed(pm1.run(module))) {
            logger.error("Phase 1 (RelAlg → SubOp) failed!");
            module.dump();
            return false;
        }
        logger.notice("Phase 1 completed successfully");
        
        // Check what's inside ExecutionGroupOp
        module.walk([&](pgx_lower::compiler::dialect::subop::ExecutionGroupOp execGroup) {
            logger.notice("Found ExecutionGroupOp");
            logger.notice("Number of regions: " + std::to_string(execGroup->getNumRegions()));
            if (execGroup.getRegion().empty()) {
                logger.error("ExecutionGroupOp has empty region!");
            } else {
                logger.notice("ExecutionGroupOp region has " + std::to_string(execGroup.getRegion().getBlocks().size()) + " blocks");
                for (auto& block : execGroup.getRegion()) {
                    logger.notice("Block has " + std::to_string(block.getOperations().size()) + " operations");
                    for (auto& op : block) {
                        logger.notice("  Operation: " + op.getName().getStringRef().str());
                        if (op.getName().getStringRef() == "subop.execution_step") {
                            logger.notice("    Found ExecutionStepOp!");
                        }
                    }
                }
            }
        });
    }
    
    // Phase 2: SubOp optimization passes (following LingoDB's Execution.cpp pattern)
    logger.notice("Phase 2: SubOp optimization passes");
    
    // Test each pass individually to identify the failing one
    std::vector<std::pair<std::string, std::function<std::unique_ptr<mlir::Pass>()>>> passes = {
        {"GlobalOpt", []() { return pgx_lower::compiler::dialect::subop::createGlobalOptPass(); }},
        {"FoldColumns", []() { return pgx_lower::compiler::dialect::subop::createFoldColumnsPass(); }},
        {"ReuseLocal", []() { return pgx_lower::compiler::dialect::subop::createReuseLocalPass(); }},
        {"SpecializeSubOp", []() { return pgx_lower::compiler::dialect::subop::createSpecializeSubOpPass(true); }},
        {"NormalizeSubOp", []() { return pgx_lower::compiler::dialect::subop::createNormalizeSubOpPass(); }},
        {"PullGatherUp", []() { return pgx_lower::compiler::dialect::subop::createPullGatherUpPass(); }},
        {"EnforceOrder", []() { return pgx_lower::compiler::dialect::subop::createEnforceOrderPass(); }},
        {"InlineNestedMap", []() { return pgx_lower::compiler::dialect::subop::createInlineNestedMapPass(); }},
        {"Finalize", []() { return pgx_lower::compiler::dialect::subop::createFinalizePass(); }},
        {"SplitIntoExecutionSteps", []() { return pgx_lower::compiler::dialect::subop::createSplitIntoExecutionStepsPass(); }},
        {"SpecializeParallel", []() { return pgx_lower::compiler::dialect::subop::createSpecializeParallelPass(); }},
        {"PrepareLowering", []() { return pgx_lower::compiler::dialect::subop::createPrepareLoweringPass(); }},
    };
    
    for (auto& [passName, passCreator] : passes) {
        logger.notice("Running SubOp pass: " + passName);
        
        // Dump module before SplitIntoExecutionSteps
        if (passName == "SplitIntoExecutionSteps") {
            logger.notice("=== Module state before SplitIntoExecutionSteps ===");
            module.walk([&](pgx_lower::compiler::dialect::subop::ExecutionGroupOp execGroup) {
                logger.notice("ExecutionGroupOp found:");
                execGroup.print(llvm::errs());
                llvm::errs() << "\n";
                logger.notice("ExecutionGroupOp details:");
                logger.notice("  Has regions: " + std::to_string(execGroup->getNumRegions()));
                if (!execGroup.getRegion().empty()) {
                    logger.notice("  Region has " + std::to_string(execGroup.getRegion().getBlocks().size()) + " blocks");
                    for (auto& block : execGroup.getRegion()) {
                        logger.notice("  Block has " + std::to_string(block.getOperations().size()) + " operations");
                        for (auto& op : block) {
                            std::string opStr;
                            llvm::raw_string_ostream os(opStr);
                            op.print(os);
                            os.flush();
                            logger.notice("    Op: " + opStr);
                        }
                    }
                }
            });
        }
        
        auto pm = mlir::PassManager(&context);
        if (passName == "Parallelize") {
            pm.addNestedPass<mlir::func::FuncOp>(pgx_lower::compiler::dialect::subop::createParallelizePass());
        } else {
            pm.addPass(passCreator());
        }
        
        if (failed(pm.run(module))) {
            logger.error("SubOp pass FAILED: " + passName);
            module.dump();
            return false;
        }
        logger.notice("SubOp pass completed: " + passName);
    }
    
    // Add Parallelize pass separately (needs nested pass)
    logger.notice("Running SubOp pass: Parallelize");
    {
        auto pm = mlir::PassManager(&context);
        pm.addNestedPass<mlir::func::FuncOp>(pgx_lower::compiler::dialect::subop::createParallelizePass());
        if (failed(pm.run(module))) {
            logger.error("SubOp pass FAILED: Parallelize");
            module.dump();
            return false;
        }
        logger.notice("SubOp pass completed: Parallelize");
    }
    
    logger.notice("Phase 2 completed successfully");
    
    // Phase 3: SubOp → Control Flow lowering (following LingoDB's Execution.cpp pattern)
    logger.notice("Phase 3: SubOp → Control Flow lowering");
    
    // Debug: Check module state before SubOp lowering
    logger.notice("=== Module state before SubOp → Control Flow ===");
    module.walk([&](pgx_lower::compiler::dialect::subop::ExecutionGroupOp execGroup) {
        logger.notice("Found ExecutionGroupOp before lowering");
        if (!execGroup.getRegion().empty()) {
            logger.notice("ExecutionGroupOp has " + std::to_string(execGroup.getRegion().getBlocks().size()) + " blocks");
            for (auto& block : execGroup.getRegion()) {
                logger.notice("Block has " + std::to_string(block.getOperations().size()) + " operations");
            }
        }
    });
    
    {
        auto pm3 = mlir::PassManager(&context);
        pgx_lower::compiler::dialect::subop::setCompressionEnabled(false);
        // Switch back to minimal pass that works, then gradually add functionality
        logger.notice("Creating MinimalSubOpToControlFlowPass - works reliably...");
        auto minimalPass = pgx_lower::compiler::dialect::subop::createMinimalSubOpToControlFlowPass();
        if (!minimalPass) {
            logger.error("createMinimalSubOpToControlFlowPass returned null!");
            return false;
        }
        logger.notice("MinimalSubOpToControlFlowPass created successfully");
        pm3.addPass(std::move(minimalPass));
        pm3.addPass(mlir::createCanonicalizerPass());
        pm3.addPass(mlir::createCSEPass());
        
        logger.notice("About to run Phase 3 pass manager...");
        
        // Verify module is valid before Phase 3
        if (failed(mlir::verify(module))) {
            logger.error("Module verification failed BEFORE Phase 3!");
            module.dump();
            return false;
        }
        logger.notice("Module verification passed before Phase 3");
        
        try {
            logger.notice("Running pm3.run(module)...");
            if (failed(pm3.run(module))) {
                logger.error("Phase 3 (SubOp → Control Flow) failed during execution!");
                logger.error("This means the pass manager run returned failure");
                
                // Check if module is still valid after failure
                if (failed(mlir::verify(module))) {
                    logger.error("Module became invalid during Phase 3 execution!");
                    module.dump();
                } else {
                    logger.notice("Module is still valid after Phase 3 failure");
                }
                return false;
            }
        } catch (const std::exception& e) {
            logger.error("Phase 3 (SubOp → Control Flow) threw exception: " + std::string(e.what()));
            return false;
        } catch (...) {
            logger.error("Phase 3 (SubOp → Control Flow) threw unknown exception!");
            logger.error("This suggests a crash during pass execution");
            return false;
        }
        logger.notice("Phase 3 pass manager completed successfully");
        logger.notice("Phase 3 completed successfully");
    }
    
    // Phase 4: DB lowering pipeline  
    logger.notice("Phase 4: DB lowering pipeline");
    {
        // Skip DB lowering since MinimalSubOpToControlFlow generates direct PostgreSQL calls
        // The DB lowering passes expect DB dialect operations, but we removed all SubOp operations
        // and generate direct func.func calls to PostgreSQL runtime functions
        logger.notice("Skipping DB lowering - MinimalSubOpToControlFlow generates direct PostgreSQL runtime calls");
        logger.notice("Phase 4 completed successfully (skipped)");
    }
    
    // Note: We skip Arrow lowering since we're using PostgreSQL instead
    
    // Now lower to LLVM
    auto pm2 = mlir::PassManager(&context);
    pm2.addPass(mlir::createConvertSCFToCFPass());
    pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm2.addPass(mlir::createArithToLLVMConversionPass());
    pm2.addPass(mlir::createConvertFuncToLLVMPass());
    
    // Reconcile unrealized casts AFTER all dialect-to-LLVM conversions (LingoDB pattern)
    pm2.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (failed(pm2.run(module))) {
        logger.error("Remaining lowering passes failed");
        logger.error("Dumping module after standard lowering failure:");
        module.dump();
        return false;
    }
    logger.notice("Lowered PostgreSQL typed field access MLIR to LLVM dialect!");
    
    // Main function should have been created by real SubOpToControlFlow pass
    // and lowered to LLVM dialect by ConvertFuncToLLVM pass
    auto mainFuncOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
    if (!mainFuncOp) {
        logger.error("Main function not found after lowering passes!");
        logger.error("Real SubOpToControlFlow should have created it");
        return false;
    }
    
    // Dump the MLIR after all lowering passes to see what's causing LLVM IR translation to fail
    logger.notice("MLIR after all lowering passes:");
    std::string mlirString;
    llvm::raw_string_ostream stream(mlirString);
    module.print(stream);
    stream.flush();
    logger.notice("Lowered MLIR: " + mlirString);

    // Enhanced ExecutionEngine creation with detailed error diagnostics
    logger.notice("Creating ExecutionEngine with enhanced error reporting...");
    
    // Test LLVM IR translation manually first to catch translation errors
    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    logger.notice("Attempting MLIR to LLVM IR translation...");
    
    auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
    if (!llvmModule) {
        logger.error("MLIR to LLVM IR translation failed - this is the root cause");
        logger.error("Check for unsupported operations or type conversion issues in the MLIR module");
        module.dump();
        return false;
    }
    logger.notice("MLIR to LLVM IR translation successful");
    
    // Verify the generated LLVM IR module
    std::string verifyErrors;
    llvm::raw_string_ostream verifyStream(verifyErrors);
    if (llvm::verifyModule(*llvmModule, &verifyStream)) {
        verifyStream.flush();
        logger.error("LLVM IR module verification failed:");
        logger.error("Verification errors: " + verifyErrors);
        llvmModule->print(llvm::errs(), nullptr);
        return false;
    }
    logger.notice("LLVM IR module verification passed");

    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto engineOptions = mlir::ExecutionEngineOptions();
    engineOptions.transformer = optPipeline;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    
    logger.notice("Attempting ExecutionEngine creation...");
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        auto errMsg = std::string();
        auto errStream = llvm::raw_string_ostream(errMsg);
        errStream << maybeEngine.takeError();
        logger.error("ExecutionEngine creation error details: " + errMsg);
        
        auto error = pgx_lower::ErrorManager::compilationError("Failed to create MLIR ExecutionEngine for typed field "
                                                               "access",
                                                               "ExecutionEngine creation failed: " + errMsg);
        pgx_lower::ErrorManager::reportError(error);
        logger.error("Failed to create MLIR ExecutionEngine for typed field access");
        return false;
    }
    logger.notice("Created MLIR ExecutionEngine for PostgreSQL typed field access!");
    auto engine = std::move(*maybeEngine);

    // Register runtime function symbols with the ExecutionEngine
    logger.notice("Registering runtime function symbols with ExecutionEngine...");
    
    // Register runtime symbols using the MLIR ExecutionEngine registerSymbols API
    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        
        // Register interface functions that the JIT code will call
        auto addSymbol = [&](const char* name, void* ptr) {
            const auto addr = llvm::orc::ExecutorAddr::fromPtr(ptr);
            symbolMap[interner(name)] = llvm::orc::ExecutorSymbolDef(addr, llvm::JITSymbolFlags::Exported);
        };
        
        addSymbol("open_postgres_table", reinterpret_cast<void*>(open_postgres_table));
        addSymbol("read_next_tuple_from_table", reinterpret_cast<void*>(read_next_tuple_from_table));
        addSymbol("close_postgres_table", reinterpret_cast<void*>(close_postgres_table));
        addSymbol("add_tuple_to_result", reinterpret_cast<void*>(add_tuple_to_result));
        addSymbol("get_int_field_mlir", reinterpret_cast<void*>(get_int_field_mlir));
        addSymbol("get_text_field", reinterpret_cast<void*>(get_text_field));
        addSymbol("get_numeric_field", reinterpret_cast<void*>(get_numeric_field));
        addSymbol("store_bool_result", reinterpret_cast<void*>(store_bool_result));
        addSymbol("store_int_result", reinterpret_cast<void*>(store_int_result));
        addSymbol("store_bigint_result", reinterpret_cast<void*>(store_bigint_result));
        addSymbol("store_text_result", reinterpret_cast<void*>(store_text_result));
        addSymbol("prepare_computed_results", reinterpret_cast<void*>(prepare_computed_results));
        addSymbol("mark_results_ready_for_streaming", reinterpret_cast<void*>(mark_results_ready_for_streaming));
        addSymbol("DataSource_get", reinterpret_cast<void*>(DataSource_get));
        // sum_aggregate removed - now implemented as pure MLIR operations
        
        return symbolMap;
    });
    
    logger.notice("Runtime function symbols registered successfully");

    // Lookup and invoke the main function
    auto mainFuncPtr = engine->lookupPacked("main");
    if (!mainFuncPtr) {
        auto error = pgx_lower::ErrorManager::compilationError("Failed to lookup main function in MLIR ExecutionEngine",
                                                               "Main function not found");
        pgx_lower::ErrorManager::reportError(error);
        logger.error("Failed to lookup main function in MLIR ExecutionEngine");
        return false;
    }

    logger.notice("Looking up main function in MLIR ExecutionEngine...");
    
    logger.notice("About to execute JIT function - this is where PostgreSQL may crash...");
    logger.notice("all compilation stages working! only jit execution remaining...");

    // Add detailed logging for JIT execution debugging
    logger.notice("JIT function pointer address: " + std::to_string(reinterpret_cast<uint64_t>(mainFuncPtr.get())));
    
    // Wrap JIT function execution in error handling to prevent server crash
    try {
        logger.notice("Calling JIT function now...");
        logger.notice("About to invoke packed function - this is the actual JIT call");
        
        // Use the packed interface which is safer for MLIR
        auto result = engine->invokePacked("main");
        if (result) {
            auto errMsg = std::string();
            auto errStream = llvm::raw_string_ostream(errMsg);
            llvm::handleAllErrors(std::move(result), [&](const llvm::ErrorInfoBase &E) {
                errStream << E.message();
            });
            logger.error("Failed to invoke main function via invokePacked: " + errMsg);
            return false;
        }
        logger.notice("JIT function invokePacked returned successfully");
        logger.notice("complete success! mlir jit function executed successfully!");
        logger.notice("all stages working: mlir compilation + jit execution!");
        logger.notice("About to return from JIT execution try block...");
        
        // Add memory barrier to ensure all writes are visible
        std::atomic_thread_fence(std::memory_order_seq_cst);
        logger.notice("Memory barrier completed after JIT execution");
    } catch (const std::exception& e) {
        logger.error("JIT function execution failed with exception: " + std::string(e.what()));
        logger.error("This indicates a runtime function is causing a crash");
        return false;
    } catch (...) {
        logger.error("JIT function execution failed with unknown exception");
        logger.error("This likely indicates a segfault in a runtime function");
        return false;
    }

    logger.notice("execute_mlir_module returning true - execution completed successfully");
    logger.notice("About to destroy ExecutionEngine...");
    
    // Add a delay to see if crash happens before destructor
    logger.notice("Adding small delay before destroying ExecutionEngine...");
    // Force a flush to ensure all logs are written
    fflush(stdout);
    fflush(stderr);
    
    // Explicitly reset the engine to trigger destructor
    engine.reset();
    logger.notice("ExecutionEngine destroyed successfully");
    return true;
}


#ifndef BUILDING_UNIT_TESTS
bool run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) {
    if (!plannedStmt) {
        logger.error("PlannedStmt is null");
        return false;
    }
    
    mlir::MLIRContext context;
    
    // Load all required dialects into the context
    context.getOrLoadDialect<pgx_lower::compiler::dialect::relalg::RelAlgDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::subop::SubOperatorDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::db::DBDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::dsa::DSADialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::util::UtilDialect>();
    context.getOrLoadDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    
    logger.debug("Using PostgreSQL AST translation for query processing");
    
    // Create the PostgreSQL AST translator
    const auto translator = postgresql_ast::createPostgreSQLASTTranslator(context, logger);
    
    // Translate the PostgreSQL AST to MLIR
    const auto module = translator->translateQuery(plannedStmt);
    
    if (!module) {
        logger.error("Failed to translate PostgreSQL AST to MLIR");
        return false;
    }
    
    // Print the generated MLIR
    logger.notice("Generated MLIR from PostgreSQL AST:");
    if (!module) {
        logger.error("Module pointer is null!");
        return false;
    }
    
    logger.notice("Module pointer is valid, attempting to verify...");
    
    // First verify the module is valid
    if (failed(mlir::verify(*module))) {
        logger.error("Module verification failed! Module is invalid.");
        return false;
    }
    logger.notice("Module verification passed");
    
    // Count operations in the module
    int opCount = 0;
    module->walk([&](mlir::Operation* op) {
        opCount++;
        logger.notice("Operation " + std::to_string(opCount) + ": " + 
                     op->getName().getStringRef().str());
    });
    logger.notice("Module contains " + std::to_string(opCount) + " operations");
    
    // Try to print operations individually
    logger.notice("Trying to print operations individually...");
    int printedOps = 0;
    module->walk([&](mlir::Operation* op) {
        printedOps++;
        try {
            logger.notice("Printing operation " + std::to_string(printedOps) + "...");
            std::string opStr;
            llvm::raw_string_ostream os(opStr);
            op->print(os);
            os.flush();
            logger.notice("Op " + std::to_string(printedOps) + ": " + opStr);
        } catch (...) {
            logger.error("Failed to print operation " + std::to_string(printedOps));
        }
    });
    
    // Skip full module dump for now since it crashes
    logger.notice("Skipping full module dump due to crash issue");
    
    // Execute the MLIR module
    bool result = executeMLIRModule(*module, logger);
    logger.notice("run_mlir_postgres_ast_translation: executeMLIRModule returned " + std::string(result ? "true" : "false"));
    logger.notice("run_mlir_postgres_ast_translation: About to return to my_executor.cpp...");
    return result;
}
#else
// Unit test stub implementation
bool run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) {
    logger.notice("PostgreSQL AST translation not available in unit tests");
    return false;
}
#endif


} // namespace mlir_runner