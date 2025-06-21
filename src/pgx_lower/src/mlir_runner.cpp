#include "mlir_runner.h"
#include "mlir_logger.h"

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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <sstream>

namespace mlir_runner {

void registerConversionPipeline() {
    mlir::PassPipelineRegistration<>(
        "convert-to-llvm", "Convert MLIR to LLVM dialect",
        [](mlir::OpPassManager &pm) {
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createConvertSCFToCFPass());
        });
}

auto run_mlir_core(int64_t intValue, MLIRLogger& logger) -> bool {
    // Create MLIR context and builder
    mlir::MLIRContext context;
    
    std::ostringstream oss;
    oss << "MLIRContext symbol address: " << (void *)&context;
    logger.debug(oss.str());
    
    // Register required dialects
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    registerConversionPipeline();

    std::unique_ptr<mlir::ExecutionEngine> engine;
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    mlir::Location loc = builder.getUnknownLoc();

    // Create a function that returns an int
    auto funcType = builder.getFunctionType({}, {builder.getI64Type()});
    auto func = mlir::func::FuncOp::create(loc, "main", funcType);
    func.setPublic();  // public so it can be called from JIT
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    auto constOp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(intValue));
    // Return the constant value
    builder.create<mlir::func::ReturnOp>(loc, constOp.getResult());
    module.push_back(func);

    if (mlir::failed(mlir::verify(module))) {
        logger.error("MLIR module verification failed");
        return false;
    }

    logger.notice("Generated MLIR program:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR: " + mlirStr);

    mlir::PassManager pm(&context);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    
    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to lower MLIR module to LLVM dialect");
        return false;
    }
    logger.notice("Lowered MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        logger.error("Failed to create MLIR ExecutionEngine");
        return false;
    }
    logger.notice("Created MLIR ExecutionEngine!");
    engine = std::move(*maybeEngine);

    auto expectedFPtr = engine->lookup("main");
    if (!expectedFPtr) {
        std::string errMsg;
        llvm::raw_string_ostream errStream(errMsg);
        errStream << expectedFPtr.takeError();
        logger.error("Failed to lookup function: " + errMsg);
        return false;
    }
    
    auto fptr = reinterpret_cast<int64_t(*)()>(*expectedFPtr);
    int64_t result = fptr();
    logger.notice("Invoked MLIR JIT-compiled function!");
    
    oss.str("");
    oss << "MLIR JIT returned: " << result;
    logger.notice(oss.str());

    return true;
}

#ifndef POSTGRESQL_EXTENSION
bool run_mlir_test(int64_t intValue) {
    ConsoleLogger logger;
    return run_mlir_core(intValue, logger);
}
#endif

}