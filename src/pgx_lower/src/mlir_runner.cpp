#include "mlir_runner.h"
#include "mlir_logger.h"

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

void registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("convert-to-llvm", "Convert MLIR to LLVM dialect", [](mlir::OpPassManager& pm) {
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertSCFToCFPass());
    });
}

extern "C" void* open_postgres_table(const char* tableName);
extern "C" int64_t read_next_tuple_from_table(void* tableHandle);
extern "C" void close_postgres_table(void* tableHandle);
extern "C" bool add_tuple_to_result(int64_t value);

bool run_mlir_postgres_table_scan(const char* tableName, MLIRLogger& logger) {
    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "Scanning PostgreSQL table '" << tableName << "' directly in MLIR JIT";
    logger.debug(oss.str());

    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

    registerConversionPipeline();

    std::unique_ptr<mlir::ExecutionEngine> engine;
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    mlir::Location loc = builder.getUnknownLoc();

    auto ptrType = builder.getI64Type();
    auto stringPtrType = builder.getI64Type();

    auto openTableType = builder.getFunctionType({stringPtrType}, {ptrType});
    auto openTableFunc = mlir::func::FuncOp::create(loc, "open_postgres_table", openTableType);
    openTableFunc.setPrivate();
    module.push_back(openTableFunc);

    auto readTupleType = builder.getFunctionType({ptrType}, {builder.getI64Type()});
    auto readTupleFunc = mlir::func::FuncOp::create(loc, "read_next_tuple_from_table", readTupleType);
    readTupleFunc.setPrivate();
    module.push_back(readTupleFunc);

    auto closeTableType = builder.getFunctionType({ptrType}, {});
    auto closeTableFunc = mlir::func::FuncOp::create(loc, "close_postgres_table", closeTableType);
    closeTableFunc.setPrivate();
    module.push_back(closeTableFunc);

    auto addTupleType = builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()});
    auto addTupleFunc = mlir::func::FuncOp::create(loc, "add_tuple_to_result", addTupleType);
    addTupleFunc.setPrivate();
    module.push_back(addTupleFunc);

    auto mainFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto mainFunc = mlir::func::FuncOp::create(loc, "main", mainFuncType);
    mainFunc.setPublic();
    mlir::Block* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto zeroConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    auto negTwoConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(-2)); // End of table
                                                                                                    // marker
    auto tableNamePtr =
        builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(reinterpret_cast<int64_t>(tableName)));

    auto openCall = builder.create<mlir::func::CallOp>(loc, openTableFunc, mlir::ValueRange{tableNamePtr});
    mlir::Value tableHandle = openCall.getResult(0);

    // Use proper scf.while loop to read entire table until end-of-table marker (-2)
    auto oneConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(true));
    
    llvm::SmallVector<mlir::Value> initialArgs;
    initialArgs.push_back(oneConst);
    initialArgs.push_back(zeroConst);
    llvm::SmallVector<mlir::Type> argTypes = {builder.getI1Type(), builder.getI64Type()};
    
    auto whileOp = builder.create<mlir::scf::WhileOp>(loc, argTypes, initialArgs);
    
    // Before region: check if we should continue reading
    auto& beforeRegion = whileOp.getBefore();
    builder.createBlock(&beforeRegion, beforeRegion.end(), argTypes, 
                       {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&beforeRegion.front());
    
    mlir::Value continueFlag = beforeRegion.front().getArgument(0);
    mlir::Value currentSum = beforeRegion.front().getArgument(1);
    
    // Always continue if the flag is true (we'll update this in the after region)
    builder.create<mlir::scf::ConditionOp>(loc, continueFlag, beforeRegion.front().getArguments());
    
    auto& afterRegion = whileOp.getAfter();
    builder.createBlock(&afterRegion, afterRegion.end(), argTypes,
                       {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&afterRegion.front());
    
    mlir::Value tupleCount = afterRegion.front().getArgument(1);
    
    auto readCall = builder.create<mlir::func::CallOp>(loc, readTupleFunc, mlir::ValueRange{tableHandle});
    mlir::Value tupleValue = readCall.getResult(0);
    
    auto isEndOfTable = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                                           tupleValue, negTwoConst);
    
    auto ifOp = builder.create<mlir::scf::IfOp>(loc,
                                               llvm::ArrayRef<mlir::Type>{builder.getI1Type(), builder.getI64Type()},
                                               isEndOfTable, true);
    
    auto& thenRegion = ifOp.getThenRegion();
    if (thenRegion.empty()) {
        builder.createBlock(&thenRegion);
    }
    builder.setInsertionPointToStart(&thenRegion.front());
    auto falseConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(false));
    llvm::SmallVector<mlir::Value> thenYieldValues;
    thenYieldValues.push_back(falseConst);
    thenYieldValues.push_back(tupleCount);
    builder.create<mlir::scf::YieldOp>(loc, thenYieldValues);
    
    auto& elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
        builder.createBlock(&elseRegion);
    }
    builder.setInsertionPointToStart(&elseRegion.front());
    auto trueConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(true));
    
    auto addTupleCall = builder.create<mlir::func::CallOp>(loc, addTupleFunc, mlir::ValueRange{tupleValue});
    
    auto oneIntConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
    auto newCount = builder.create<mlir::arith::AddIOp>(loc, tupleCount, oneIntConst);
    
    llvm::SmallVector<mlir::Value> elseYieldValues;
    elseYieldValues.push_back(trueConst);
    elseYieldValues.push_back(newCount.getResult());
    builder.create<mlir::scf::YieldOp>(loc, elseYieldValues);
    
    // Yield the results back to the while condition
    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::scf::YieldOp>(loc, ifOp.getResults());
    
    // Continue after the while loop
    builder.setInsertionPointAfter(whileOp);

    mlir::Value finalCount = whileOp.getResult(1);

    builder.create<mlir::func::CallOp>(loc, closeTableFunc, mlir::ValueRange{tableHandle});

    builder.create<mlir::func::ReturnOp>(loc, finalCount);

    module.push_back(mainFunc);

    if (mlir::failed(mlir::verify(module))) {
        logger.error("MLIR module verification failed for PostgreSQL table scan");
        return false;
    }

    logger.notice("Generated MLIR program for PostgreSQL table scan:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR: " + mlirStr);

    mlir::PassManager pm(&context);
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to lower MLIR module to LLVM dialect");
        return false;
    }
    logger.notice("Lowered PostgreSQL table scan MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context.appendDialectRegistry(registry);

    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        logger.error("Failed to create MLIR ExecutionEngine for table scan");
        return false;
    }
    logger.notice("Created MLIR ExecutionEngine for PostgreSQL table scan!");
    engine = std::move(*maybeEngine);

    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        symbolMap[interner("open_postgres_table")] =
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&open_postgres_table)),
                                         llvm::JITSymbolFlags::Exported);
        symbolMap[interner("read_next_tuple_from_table")] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&read_next_tuple_from_table)),
            llvm::JITSymbolFlags::Exported);
        symbolMap[interner("close_postgres_table")] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&close_postgres_table)),
            llvm::JITSymbolFlags::Exported);
        symbolMap[interner("add_tuple_to_result")] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&add_tuple_to_result)),
            llvm::JITSymbolFlags::Exported);
        return symbolMap;
    });

    auto expectedFPtr = engine->lookup("main");
    if (!expectedFPtr) {
        std::string errMsg;
        llvm::raw_string_ostream errStream(errMsg);
        errStream << expectedFPtr.takeError();
        logger.error("Failed to lookup function: " + errMsg);
        return false;
    }

    auto fptr = reinterpret_cast<int64_t (*)()>(*expectedFPtr);
    int64_t result = fptr();
    logger.notice("Invoked MLIR JIT PostgreSQL table scanner!");

    oss.str("");
    oss << "PostgreSQL table scan completed with sum: " << result;
    logger.notice(oss.str());

    return true;
}

} // namespace mlir_runner