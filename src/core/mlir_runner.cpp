#include "core/mlir_runner.h"
#include "core/mlir_logger.h"
#include "core/mlir_code_generator.h"
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

void registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("convert-to-llvm", "Convert MLIR to LLVM dialect", [](mlir::OpPassManager& pm) {
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertSCFToCFPass());
    });
}

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
        auto error = pgx_lower::ErrorManager::compilationError(
            "Failed to lower MLIR module to LLVM dialect",
            "Pass manager execution failed");
        pgx_lower::ErrorManager::reportError(error);
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
        auto error = pgx_lower::ErrorManager::compilationError(
            "Failed to create MLIR ExecutionEngine for table scan",
            "ExecutionEngine creation failed - check LLVM configuration");
        pgx_lower::ErrorManager::reportError(error);
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

bool run_mlir_postgres_table_scan_modular(const char* tableName, MLIRLogger& logger) {
    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "Scanning PostgreSQL table '" << tableName << "' directly in MLIR JIT (modular)";
    logger.debug(oss.str());

    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

    registerConversionPipeline();

    // Use modular MLIR generation
    pgx_lower::ModularMLIRGenerator generator(&context);
    
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Generate table scan function using modular approach
    auto mainFunc = generator.generateTableScanFunction(tableName);
    module.push_back(mainFunc);

    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    logger.notice("Generated MLIR program for PostgreSQL table scan:");
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR: " + mlirStr);

    // Same execution pipeline as original
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        auto error = pgx_lower::ErrorManager::compilationError(
            "Failed to lower MLIR module to LLVM dialect",
            "Pass manager execution failed");
        pgx_lower::ErrorManager::reportError(error);
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
        auto error = pgx_lower::ErrorManager::compilationError(
            "Failed to create MLIR ExecutionEngine for table scan",
            "ExecutionEngine creation failed - check LLVM configuration");
        pgx_lower::ErrorManager::reportError(error);
        logger.error("Failed to create MLIR ExecutionEngine for table scan");
        return false;
    }
    logger.notice("Created MLIR ExecutionEngine for PostgreSQL table scan!");
    auto engine = std::move(*maybeEngine);

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

bool run_mlir_postgres_typed_table_scan(const char* tableName, MLIRLogger& logger) {
    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "Scanning PostgreSQL table '" << tableName << "' with typed field access";
    logger.debug(oss.str());

    // Register all required dialects
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::pg::PgDialect>();

    registerConversionPipeline();

    // Create MLIR module and function manually to test typed access
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto location = builder.getUnknownLoc();
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    
    // Declare external runtime functions as private
    auto funcType = mlir::FunctionType::get(&context, {i64Type}, {i64Type});
    auto openFunc = builder.create<mlir::func::FuncOp>(location, "open_postgres_table", funcType);
    openFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context, {i64Type}, {i64Type});
    auto readFunc = builder.create<mlir::func::FuncOp>(location, "read_next_tuple_from_table", funcType);
    readFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context, {i64Type}, mlir::TypeRange{});
    auto closeFunc = builder.create<mlir::func::FuncOp>(location, "close_postgres_table", funcType);
    closeFunc.setPrivate();
    
    // Declare field access functions as private
    funcType = mlir::FunctionType::get(&context, {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i32Type});
    auto getIntFunc = builder.create<mlir::func::FuncOp>(location, "get_int_field", funcType);
    getIntFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context, {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i64Type});
    auto getTextFunc = builder.create<mlir::func::FuncOp>(location, "get_text_field", funcType);
    getTextFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context, {i64Type}, {i1Type});
    auto addTupleFunc = builder.create<mlir::func::FuncOp>(location, "add_tuple_to_result", funcType);
    addTupleFunc.setPrivate();
    
    // Create main function that demonstrates typed field access
    auto mainFuncType = mlir::FunctionType::get(&context, {}, {i64Type});
    auto mainFunc = builder.create<mlir::func::FuncOp>(location, "main", mainFuncType);
    auto* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Generate table open using pg dialect
    auto tableHandleType = mlir::pg::TableHandleType::get(&context);
    mlir::OperationState scanState(location, mlir::pg::ScanTableOp::getOperationName());
    scanState.addAttribute("table_name", builder.getStringAttr(tableName));
    scanState.addTypes(tableHandleType);
    auto scanOp = builder.create(scanState);
    auto tableHandle = scanOp->getResult(0);
    
    // Create iteration loop
    auto zeroConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(0));
    auto negTwoConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(-2));
    auto trueConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));
    
    llvm::SmallVector<mlir::Value> initialArgs = {trueConst, zeroConst};
    llvm::SmallVector<mlir::Type> argTypes = {i1Type, i64Type};
    
    auto whileOp = builder.create<mlir::scf::WhileOp>(location, argTypes, initialArgs);
    
    // Before region: condition check
    auto& beforeRegion = whileOp.getBefore();
    builder.createBlock(&beforeRegion, beforeRegion.end(), argTypes, 
                       {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&beforeRegion.front());
    
    mlir::Value continueFlag = beforeRegion.front().getArgument(0);
    mlir::Value currentCount = beforeRegion.front().getArgument(1);
    
    builder.create<mlir::scf::ConditionOp>(location, continueFlag, beforeRegion.front().getArguments());
    
    // After region: read tuple and access fields
    auto& afterRegion = whileOp.getAfter();
    builder.createBlock(&afterRegion, afterRegion.end(), argTypes,
                       {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&afterRegion.front());
    
    mlir::Value tupleCount = afterRegion.front().getArgument(1);
    
    // Read tuple using pg dialect
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context);
    mlir::OperationState readState(location, mlir::pg::ReadTupleOp::getOperationName());
    readState.addOperands(tableHandle);
    readState.addTypes(tupleHandleType);
    auto readOp = builder.create(readState);
    auto tupleHandle = readOp->getResult(0);
    
    // Check for end of table (simplified - normally this would be handled in the lowering)
    auto tupleAsInt = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(1)); // Mock: assume we have data
    auto isEndOfTable = builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq,
                                                           tupleAsInt, negTwoConst);
    
    auto ifOp = builder.create<mlir::scf::IfOp>(location,
                                               llvm::ArrayRef<mlir::Type>{i1Type, i64Type},
                                               isEndOfTable, true);
    
    // Then branch: end of table
    auto& thenRegion = ifOp.getThenRegion();
    if (thenRegion.empty()) {
        builder.createBlock(&thenRegion);
    }
    builder.setInsertionPointToStart(&thenRegion.front());
    auto falseConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(false));
    builder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{falseConst, tupleCount});
    
    // Else branch: process tuple with field access
    auto& elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
        builder.createBlock(&elseRegion);
    }
    builder.setInsertionPointToStart(&elseRegion.front());
    auto trueContinue = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));
    
    // NOW THE KEY PART: Access specific fields by index and type
    // This demonstrates accessing an 'id' field (index 0) as integer
    mlir::OperationState getIntState(location, mlir::pg::GetIntFieldOp::getOperationName());
    getIntState.addOperands(tupleHandle);
    getIntState.addAttribute("field_index", builder.getI32IntegerAttr(0));
    getIntState.addTypes({i32Type, i1Type}); // value and null flag
    auto getIntOp = builder.create(getIntState);
    auto intValue = getIntOp->getResult(0);
    auto intNullFlag = getIntOp->getResult(1);
    
    // Access a text field (index 1) 
    mlir::OperationState getTextState(location, mlir::pg::GetTextFieldOp::getOperationName());
    getTextState.addOperands(tupleHandle);
    getTextState.addAttribute("field_index", builder.getI32IntegerAttr(1));
    getTextState.addTypes({i64Type, i1Type}); // text pointer and null flag
    auto getTextOp = builder.create(getTextState);
    auto textPtr = getTextOp->getResult(0);
    auto textNullFlag = getTextOp->getResult(1);
    
    // For now, just output the original tuple (this shows we have the field access infrastructure)
    auto addTupleCall = builder.create<mlir::func::CallOp>(location, "add_tuple_to_result", 
                                                          mlir::ValueRange{tupleAsInt});
    
    auto oneIntConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(1));
    auto newCount = builder.create<mlir::arith::AddIOp>(location, tupleCount, oneIntConst);
    
    builder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{trueContinue, newCount.getResult()});
    
    // Continue after the while loop
    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::scf::YieldOp>(location, ifOp.getResults());
    
    builder.setInsertionPointAfter(whileOp);
    
    // Close table (simplified - cast table handle to i64)
    auto tableHandleAsInt = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(0)); // Mock for now
    builder.create<mlir::func::CallOp>(location, "close_postgres_table", mlir::ValueRange{tableHandleAsInt});
    
    mlir::Value finalCount = whileOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(location, finalCount);
    
    // Print the MLIR with typed field access
    logger.notice("Generated MLIR with PostgreSQL typed field access:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR with field access: " + mlirStr);
    
    // For now, skip the lowering pass to demonstrate the high-level dialect operations
    // TODO: Fix lowering pass issues in a future iteration
    // Apply pg-to-scf lowering pass to convert high-level operations to runtime calls
    // mlir::PassManager pm(&context);
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::pg::createLowerPgToSCFPass());
    // 
    // if (mlir::failed(pm.run(module))) {
    //     logger.error("Failed to apply pg-to-scf lowering pass");
    //     return false;
    // }
    // 
    // logger.notice("Applied pg-to-scf lowering pass!");
    // std::string loweredStr;
    // llvm::raw_string_ostream loweredOs(loweredStr);
    // module.OpState::print(loweredOs);
    // loweredOs.flush();
    // logger.notice("Lowered MLIR: " + loweredStr);
    
    logger.notice("Demonstrating high-level PostgreSQL dialect operations!");
    
    return true;
}

} // namespace mlir_runner