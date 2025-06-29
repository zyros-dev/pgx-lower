#include "core/mlir_runner.h"
#include "core/mlir_builder.h"
#include "core/mlir_logger.h"
#include "core/error_handling.h"
#include "interfaces/mlir_c_interface.h"
#include "dialects/pg/PgDialect.h"
#include "dialects/pg/LowerPgToSCF.h"

#include <fstream>

// Prevent libintl.h conflicts with PostgreSQL macros
// This is a bit strange to me - so LLVM drags in some macros from libintl.h
// and those conflict with things inside of libintl.h. So this should resolve
// those problems?
#define ENABLE_NLS 0

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/TargetSelect.h"
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <sstream>
#include <cstring>

namespace mlir_runner {

// MLIR Execution Engine - handles lowering and JIT execution
bool executeMLIRModule(mlir::ModuleOp& module, MLIRLogger& logger) {
    auto& context = *module.getContext();

    // Apply pg-to-scf lowering pass to convert high-level operations to runtime calls
    auto pm = mlir::PassManager(&context);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::pg::createLowerPgToSCFPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to apply pg-to-scf lowering pass");
        return false;
    }

    logger.notice("Applied pg-to-scf lowering pass!");
    auto loweredStr = std::string();
    auto loweredOs = llvm::raw_string_ostream(loweredStr);
    module.OpState::print(loweredOs);
    loweredOs.flush();
    logger.notice("Lowered MLIR: " + loweredStr);

    // Continue with full lowering pipeline to execute
    pm.clear();
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        auto error = pgx_lower::ErrorManager::compilationError("Failed to lower MLIR module to LLVM dialect",
                                                               "Pass manager execution failed");
        pgx_lower::ErrorManager::reportError(error);
        logger.error("Failed to lower MLIR module to LLVM dialect");
        return false;
    }
    logger.notice("Lowered PostgreSQL typed field access MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context.appendDialectRegistry(registry);

    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto engineOptions = mlir::ExecutionEngineOptions();
    engineOptions.transformer = optPipeline;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        auto error = pgx_lower::ErrorManager::compilationError("Failed to create MLIR ExecutionEngine for typed field "
                                                               "access",
                                                               "ExecutionEngine creation failed - check LLVM "
                                                               "configuration");
        pgx_lower::ErrorManager::reportError(error);
        logger.error("Failed to create MLIR ExecutionEngine for typed field access");
        return false;
    }
    logger.notice("Created MLIR ExecutionEngine for PostgreSQL typed field access!");
    auto engine = std::move(*maybeEngine);

    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        auto symbolMap = llvm::orc::SymbolMap();
        symbolMap[interner("open_postgres_table")] =
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&open_postgres_table)),
                                         llvm::JITSymbolFlags::Exported);
        symbolMap[interner("read_next_tuple_from_table")] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&read_next_tuple_from_table)),
            llvm::JITSymbolFlags::Exported);
        symbolMap[interner("close_postgres_table")] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&close_postgres_table)),
            llvm::JITSymbolFlags::Exported);
        symbolMap[interner("add_tuple_to_result")] =
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&add_tuple_to_result)),
                                         llvm::JITSymbolFlags::Exported);
        symbolMap[interner("get_int_field")] =
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&get_int_field)),
                                         llvm::JITSymbolFlags::Exported);
        symbolMap[interner("get_text_field")] =
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&get_text_field)),
                                         llvm::JITSymbolFlags::Exported);
        return symbolMap;
    });

    auto expectedFPtr = engine->lookup("main");
    if (!expectedFPtr) {
        auto errMsg = std::string();
        auto errStream = llvm::raw_string_ostream(errMsg);
        errStream << expectedFPtr.takeError();
        logger.error("Failed to lookup function: " + errMsg);
        return false;
    }

    auto fptr = reinterpret_cast<int64_t (*)()>(*expectedFPtr);
    const int64_t result = fptr();
    logger.notice("Invoked MLIR JIT PostgreSQL typed field access!");

    std::ostringstream oss;
    oss << "PostgreSQL typed field access completed with result: " << result;
    logger.notice(oss.str());

    logger.notice("MLIR successfully handled the query");

    return true;
}

bool run_mlir_postgres_typed_table_scan_with_columns(const char* tableName,
                                                     const std::vector<int>& selectedColumns,
                                                     MLIRLogger& logger) {
    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "Scanning PostgreSQL table '" << tableName << "' with column subset: ";
    for (size_t i = 0; i < selectedColumns.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << selectedColumns[i];
    }
    logger.debug(oss.str());

    // Use MLIRBuilder to generate the MLIR module
    auto builder = mlir_builder::createMLIRBuilder(context);
    auto module = builder->buildTableScanModule(tableName, selectedColumns);

    if (!module) {
        logger.error("Failed to build MLIR module");
        return false;
    }

    // Print the MLIR with typed field access
    logger.notice("Generated MLIR with PostgreSQL typed field access:");
    auto mlirStr = std::string();
    auto os = llvm::raw_string_ostream(mlirStr);
    module->OpState::print(os);
    os.flush();
    logger.notice("MLIR with field access: " + mlirStr);

    // Execute the MLIR module
    return executeMLIRModule(*module, logger);
}

} // namespace mlir_runner