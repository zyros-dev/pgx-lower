#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/error_handling.h"
#include "core/postgresql_ast_translator.h"
// PG dialect removed - using RelAlg instead
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/LowerSubOpToDB.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/LowerDBToDSA.h"
#include "dialects/dsa/DSADialect.h"
#include "dialects/util/LowerDSAToLLVM.h"

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
    
    logger.notice("Using LingoDB-style lowering pipeline: RelAlg → SubOp → DB → DSA → LLVM");
    logger.notice("(PostgreSQL integration happens in the lowering passes)");
    
    // Run each pass individually to see intermediate results
    
    // RelAlg → SubOp lowering
    logger.notice("=== Running RelAlg → SubOp lowering pass ===");
    auto relalgToSubOpPM = mlir::PassManager(&context);
    relalgToSubOpPM.addPass(pgx_lower::compiler::dialect::relalg::createLowerRelAlgToSubOpPass());
    if (failed(relalgToSubOpPM.run(module))) {
        logger.error("RelAlg → SubOp lowering failed");
        return false;
    }
    logger.notice("Module after RelAlg → SubOp:");
    std::string afterPgStr;
    llvm::raw_string_ostream afterPgOs(afterPgStr);
    module.print(afterPgOs);
    afterPgOs.flush();
    logger.notice(afterPgStr);
    
    // SubOp → DB lowering
    logger.notice("=== Running SubOp → DB lowering pass ===");
    auto subOpToDbPM = mlir::PassManager(&context);
    subOpToDbPM.addPass(pgx_lower::compiler::dialect::subop::createLowerSubOpPass());
    if (failed(subOpToDbPM.run(module))) {
        logger.error("SubOp → DB lowering failed");
        return false;
    }
    logger.notice("Module after SubOp → DB:");
    std::string afterSubOpStr;
    llvm::raw_string_ostream afterSubOpOs(afterSubOpStr);
    module.print(afterSubOpOs);
    afterSubOpOs.flush();
    logger.notice(afterSubOpStr);
    
    // DB → DSA lowering
    logger.notice("=== Running DB → DSA lowering pass ===");
    auto dbToDsaPM = mlir::PassManager(&context);
    dbToDsaPM.addPass(pgx_lower::compiler::dialect::db::createLowerToStdPass());
    if (failed(dbToDsaPM.run(module))) {
        logger.error("DB → DSA lowering failed");
        return false;
    }
    logger.notice("Module after DB → DSA:");
    std::string afterDbStr;
    llvm::raw_string_ostream afterDbOs(afterDbStr);
    module.print(afterDbOs);
    afterDbOs.flush();
    logger.notice(afterDbStr);
    
    // DSA → LLVM lowering
    logger.notice("=== Running DSA → LLVM lowering pass ===");
    auto dsaToLlvmPM = mlir::PassManager(&context);
    dsaToLlvmPM.addPass(pgx_lower::compiler::dialect::util::createUtilToLLVMPass());
    if (failed(dsaToLlvmPM.run(module))) {
        logger.error("DSA → LLVM lowering failed");
        return false;
    }
    logger.notice("Module after DSA → LLVM:");
    std::string afterDsaStr;
    llvm::raw_string_ostream afterDsaOs(afterDsaStr);
    module.print(afterDsaOs);
    afterDsaOs.flush();
    logger.notice(afterDsaStr);
    
    // Reconcile unrealized casts after dialect conversions
    logger.notice("=== Running reconcile unrealized casts pass ===");
    auto reconcilePM = mlir::PassManager(&context);
    reconcilePM.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (failed(reconcilePM.run(module))) {
        logger.error("Reconcile unrealized casts failed");
        return false;
    }
    
    logger.notice("LingoDB-style lowering pipeline completed!");

    // Check for remaining dialect operations after lowering
    bool hasUnloweredOps = false;
    module.walk([&](mlir::Operation* op) {
        const std::string opName = op->getName().getStringRef().str();
        if (opName.substr(0, 3) == "pg." || 
            opName.substr(0, 6) == "subop." || 
            opName.substr(0, 3) == "db." || 
            opName.substr(0, 4) == "dsa.") {
            logger.notice("Remaining dialect operation after lowering: " + opName);
            hasUnloweredOps = true;
        }
    });
    
    if (hasUnloweredOps) {
        logger.error("ERROR: Dialect operations remain after lowering pipeline!");
        logger.error("This means the LingoDB-style lowering is incomplete.");
        // Let it fail - no fallbacks!
    }
    
    auto pm2 = mlir::PassManager(&context);
    // Follow LingoDB's proven pass order
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
        addSymbol("get_int_field", reinterpret_cast<void*>(get_int_field));
        addSymbol("get_text_field", reinterpret_cast<void*>(get_text_field));
        addSymbol("get_numeric_field", reinterpret_cast<void*>(get_numeric_field));
        addSymbol("store_bool_result", reinterpret_cast<void*>(store_bool_result));
        addSymbol("store_int_result", reinterpret_cast<void*>(store_int_result));
        addSymbol("store_bigint_result", reinterpret_cast<void*>(store_bigint_result));
        addSymbol("store_text_result", reinterpret_cast<void*>(store_text_result));
        // sum_aggregate removed - now implemented as pure MLIR operations
        
        return symbolMap;
    });
    
    logger.notice("Runtime function symbols registered successfully");

    // Lookup and invoke the main function
    auto mainFunc = engine->lookupPacked("main");
    if (!mainFunc) {
        auto error = pgx_lower::ErrorManager::compilationError("Failed to lookup main function in MLIR ExecutionEngine",
                                                               "Main function not found");
        pgx_lower::ErrorManager::reportError(error);
        logger.error("Failed to lookup main function in MLIR ExecutionEngine");
        return false;
    }

    logger.notice("Looking up main function in MLIR ExecutionEngine...");
    // Use LingoDB pattern: void (*)() function signature
    auto fptr = reinterpret_cast<void (*)()>(mainFunc.get());

    logger.notice("About to execute JIT function - this is where PostgreSQL may crash...");
    logger.notice("all compilation stages working! only jit execution remaining...");

    // Add detailed logging for JIT execution debugging
    logger.notice("JIT function pointer address: " + std::to_string(reinterpret_cast<uint64_t>(fptr)));
    
    // Wrap JIT function execution in error handling to prevent server crash
    try {
        logger.notice("Calling JIT function now...");
        fptr();  // void function call (LingoDB pattern)
        logger.notice("complete success! mlir jit function executed successfully!");
        logger.notice("all stages working: mlir compilation + jit execution!");
    } catch (const std::exception& e) {
        logger.error("JIT function execution failed with exception: " + std::string(e.what()));
        logger.error("This indicates a runtime function is causing a crash");
        return false;
    } catch (...) {
        logger.error("JIT function execution failed with unknown exception");
        logger.error("This likely indicates a segfault in a runtime function");
        return false;
    }

    return true;
}


#ifndef BUILDING_UNIT_TESTS
bool run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) {
    if (!plannedStmt) {
        logger.error("PlannedStmt is null");
        return false;
    }
    
    mlir::MLIRContext context;
    
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
    auto mlirStr = std::string();
    auto os = llvm::raw_string_ostream(mlirStr);
    module->print(os);
    os.flush();
    logger.notice("AST-generated MLIR: " + mlirStr);
    
    // Execute the MLIR module
    return executeMLIRModule(*module, logger);
}
#else
// Unit test stub implementation
bool run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) {
    logger.notice("PostgreSQL AST translation not available in unit tests");
    return false;
}
#endif


} // namespace mlir_runner