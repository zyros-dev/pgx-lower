#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/error_handling.h"
#include "core/postgresql_ast_translator.h"
#include "interfaces/mlir_c_interface.h"
#include "dialects/pg/PgDialect.h"
#include "dialects/pg/LowerPgToSCF.h"

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
#include "dialects/pg/LowerPgToSCF.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <sstream>
#include <cstring>

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

    // Apply lowering passes with detailed error reporting
    auto pm = mlir::PassManager(&context);
    pm.enableVerifier(true);
    
    logger.notice("About to create LowerPgToSCFPass...");
    auto lowerPass = mlir::pg::createLowerPgToSCFPass();
    logger.notice("AFTER createLowerPgToSCFPass() returned successfully");
    if (!lowerPass) {
        logger.error("createLowerPgToSCFPass() returned null!");
        return false;
    }
    logger.notice("LowerPgToSCFPass created, adding to PassManager...");
    
    // Add debug logging to see exactly what's happening
    logger.notice("About to add pass to PassManager...");
    pm.addPass(std::move(lowerPass));
    logger.notice("Pass added to PassManager successfully");
    
    logger.notice("Running pg-to-scf lowering pass...");
    logger.notice("About to call pm.run(module)...");
    auto passResult = pm.run(module);
    logger.notice("pm.run(module) returned");
    if (failed(passResult)) {
        logger.error("pg-to-scf lowering pass failed - passResult indicates failure");
        logger.error("Dumping module state when lowering failed:");
        module.dump();
        return false;
    }
    logger.notice("Applied pg-to-scf lowering pass!");

    // Check for remaining pg operations after lowering
    bool hasPgOps = false;
    module.walk([&](mlir::Operation* op) {
        std::string opName = op->getName().getStringRef().str();
        if (opName.substr(0, 3) == "pg.") {
            logger.error("Remaining pg operation after lowering: " + opName);
            hasPgOps = true;
        }
    });
    
    if (hasPgOps) {
        logger.error("Lowering incomplete - some pg operations remain");
        logger.notice("Dumping module after failed lowering:");
        module.dump();
        return false;
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
            auto addr = llvm::orc::ExecutorAddr::fromPtr(ptr);
            symbolMap[interner(name)] = llvm::orc::ExecutorSymbolDef(addr, llvm::JITSymbolFlags::Exported);
        };
        
        addSymbol("open_postgres_table", reinterpret_cast<void*>(open_postgres_table));
        addSymbol("read_next_tuple_from_table", reinterpret_cast<void*>(read_next_tuple_from_table));
        addSymbol("close_postgres_table", reinterpret_cast<void*>(close_postgres_table));
        addSymbol("add_tuple_to_result", reinterpret_cast<void*>(add_tuple_to_result));
        addSymbol("get_int_field", reinterpret_cast<void*>(get_int_field));
        addSymbol("get_text_field", reinterpret_cast<void*>(get_text_field));
        addSymbol("store_bool_result", reinterpret_cast<void*>(store_bool_result));
        addSymbol("store_int_result", reinterpret_cast<void*>(store_int_result));
        addSymbol("store_bigint_result", reinterpret_cast<void*>(store_bigint_result));
        
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
    logger.notice("🚀 ALL COMPILATION STAGES WORKING! Only JIT execution remaining...");
    
    // Add detailed logging for JIT execution debugging
    logger.notice("JIT function pointer address: " + std::to_string(reinterpret_cast<uint64_t>(fptr)));
    
    // Wrap JIT function execution in error handling to prevent server crash
    try {
        logger.notice("Calling JIT function now...");
        fptr();  // void function call (LingoDB pattern)
        logger.notice("🎉 COMPLETE SUCCESS! MLIR JIT function executed successfully!");
        logger.notice("🏆 ALL STAGES WORKING: MLIR compilation + JIT execution!");
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


bool run_mlir_postgres_ast_translation(PlannedStmt* plannedStmt, MLIRLogger& logger) {
    if (!plannedStmt) {
        logger.error("PlannedStmt is null");
        return false;
    }
    
    mlir::MLIRContext context;
    
    logger.debug("Using PostgreSQL AST translation for query processing");
    
    // Create the PostgreSQL AST translator
    auto translator = postgresql_ast::createPostgreSQLASTTranslator(context, logger);
    
    // Translate the PostgreSQL AST to MLIR
    auto module = translator->translateQuery(plannedStmt);
    
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


} // namespace mlir_runner