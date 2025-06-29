#include "core/mlir_runner.h"
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

    llvm::SmallVector<mlir::Value> openOperands = {tableNamePtr};
    auto openCall = builder.create<mlir::func::CallOp>(loc, openTableFunc, openOperands);
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
    builder.createBlock(&beforeRegion, beforeRegion.end(), argTypes, {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&beforeRegion.front());

    mlir::Value continueFlag = beforeRegion.front().getArgument(0);
    mlir::Value currentSum = beforeRegion.front().getArgument(1);

    // Always continue if the flag is true (we'll update this in the after region)
    builder.create<mlir::scf::ConditionOp>(loc, continueFlag, beforeRegion.front().getArguments());

    auto& afterRegion = whileOp.getAfter();
    builder.createBlock(&afterRegion, afterRegion.end(), argTypes, {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&afterRegion.front());

    mlir::Value tupleCount = afterRegion.front().getArgument(1);

    llvm::SmallVector<mlir::Value> readOperands = {tableHandle};
    auto readCall = builder.create<mlir::func::CallOp>(loc, readTupleFunc, readOperands);
    mlir::Value tupleValue = readCall.getResult(0);

    auto isEndOfTable = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tupleValue, negTwoConst);

    auto ifOp = builder.create<mlir::scf::IfOp>(loc,
                                                llvm::ArrayRef<mlir::Type>{builder.getI1Type(), builder.getI64Type()},
                                                isEndOfTable,
                                                true);

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

    llvm::SmallVector<mlir::Value> addOperands = {tupleValue};
    auto addTupleCall = builder.create<mlir::func::CallOp>(loc, addTupleFunc, addOperands);

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

    llvm::SmallVector<mlir::Value> closeOperands = {tableHandle};
    builder.create<mlir::func::CallOp>(loc, closeTableFunc, closeOperands);

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
        auto error = pgx_lower::ErrorManager::compilationError("Failed to lower MLIR module to LLVM dialect",
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
        auto error = pgx_lower::ErrorManager::compilationError("Failed to create MLIR ExecutionEngine for table scan",
                                                               "ExecutionEngine creation failed - check LLVM "
                                                               "configuration");
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
        symbolMap[interner("add_tuple_to_result")] =
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&add_tuple_to_result)),
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
    funcType =
        mlir::FunctionType::get(&context, {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i32Type});
    auto getIntFunc = builder.create<mlir::func::FuncOp>(location, "get_int_field", funcType);
    getIntFunc.setPrivate();

    funcType =
        mlir::FunctionType::get(&context, {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i64Type});
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
    builder.createBlock(&beforeRegion, beforeRegion.end(), argTypes, {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&beforeRegion.front());

    mlir::Value continueFlag = beforeRegion.front().getArgument(0);
    mlir::Value currentCount = beforeRegion.front().getArgument(1);

    builder.create<mlir::scf::ConditionOp>(location, continueFlag, beforeRegion.front().getArguments());

    // After region: read tuple and access fields
    auto& afterRegion = whileOp.getAfter();
    builder.createBlock(&afterRegion, afterRegion.end(), argTypes, {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&afterRegion.front());

    mlir::Value tupleCount = afterRegion.front().getArgument(1);

    // Read tuple using pg dialect
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context);
    mlir::OperationState readState(location, mlir::pg::ReadTupleOp::getOperationName());
    readState.addOperands(tableHandle);
    readState.addTypes(tupleHandleType);
    auto readOp = builder.create(readState);
    auto tupleHandle = readOp->getResult(0);

    // The lowering pass will convert tupleHandle to an i64 value from read_next_tuple_from_table
    // We need to compare that result with -2 to check for end of table
    // For now, we'll create a simplified version that will be fixed by the lowering pass

    // Create an unrealized conversion cast to represent the tuple as i64 for comparison
    // This will be cleaned up by the lowering pass
    auto tupleAsI64 = builder.create<mlir::UnrealizedConversionCastOp>(location, i64Type, tupleHandle).getResult(0);
    auto isEndOfTable =
        builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, tupleAsI64, negTwoConst);

    auto ifOp =
        builder.create<mlir::scf::IfOp>(location, llvm::ArrayRef<mlir::Type>{i1Type, i64Type}, isEndOfTable, true);

    // Then branch: end of table
    auto& thenRegion = ifOp.getThenRegion();
    if (thenRegion.empty()) {
        builder.createBlock(&thenRegion);
    }
    builder.setInsertionPointToStart(&thenRegion.front());
    auto falseConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(false));
    llvm::SmallVector<mlir::Value> thenYieldOperands = {falseConst, tupleCount};
    builder.create<mlir::scf::YieldOp>(location, thenYieldOperands);

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
    llvm::SmallVector<mlir::Value> addOperands = {tupleAsI64};
    auto addTupleCall = builder.create<mlir::func::CallOp>(location, addTupleFunc, addOperands);
    // Note: addTupleCall returns i1, not i64

    auto oneIntConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(1));
    auto newCount = builder.create<mlir::arith::AddIOp>(location, tupleCount, oneIntConst);

    llvm::SmallVector<mlir::Value> elseYieldOperands = {trueContinue, newCount.getResult()};
    builder.create<mlir::scf::YieldOp>(location, elseYieldOperands);

    // Continue after the while loop
    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::scf::YieldOp>(location, ifOp.getResults());

    builder.setInsertionPointAfter(whileOp);

    // Close table (simplified - cast table handle to i64)
    auto tableHandleAsInt = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(0)); // Mock for
                                                                                                             // now
    llvm::SmallVector<mlir::Value> closeOperands = {tableHandleAsInt};
    builder.create<mlir::func::CallOp>(location, closeFunc, closeOperands);

    mlir::Value finalCount = whileOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(location, finalCount);

    // Print the MLIR with typed field access
    logger.notice("Generated MLIR with PostgreSQL typed field access:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR with field access: " + mlirStr);

    // Apply pg-to-scf lowering pass to convert high-level operations to runtime calls
    mlir::PassManager pm(&context);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::pg::createLowerPgToSCFPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to apply pg-to-scf lowering pass");
        return false;
    }

    logger.notice("Applied pg-to-scf lowering pass!");
    std::string loweredStr;
    llvm::raw_string_ostream loweredOs(loweredStr);
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
    mlir::ExecutionEngineOptions engineOptions;
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
        std::string errMsg;
        llvm::raw_string_ostream errStream(errMsg);
        errStream << expectedFPtr.takeError();
        logger.error("Failed to lookup function: " + errMsg);
        return false;
    }

    auto fptr = reinterpret_cast<int64_t (*)()>(*expectedFPtr);
    int64_t result = fptr();
    logger.notice("Invoked MLIR JIT PostgreSQL typed field access!");

    oss.str("");
    oss << "PostgreSQL typed field access completed with result: " << result;
    logger.notice(oss.str());

    logger.notice("Demonstrating high-level PostgreSQL dialect operations!");

    return true;
}

bool run_mlir_postgres_typed_table_scan_with_columns(const char* tableName,
                                                     const std::vector<int>& selectedColumns,
                                                     MLIRLogger& logger) {
    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "Scanning PostgreSQL table '" << tableName << "' with column subset: ";
    for (size_t i = 0; i < selectedColumns.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << selectedColumns[i];
    }
    logger.debug(oss.str());

    // Register all required dialects
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::pg::PgDialect>();

    registerConversionPipeline();

    // Create MLIR module and function with actual column selection
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
    funcType =
        mlir::FunctionType::get(&context, {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i32Type});
    auto getIntFunc = builder.create<mlir::func::FuncOp>(location, "get_int_field", funcType);
    getIntFunc.setPrivate();

    funcType =
        mlir::FunctionType::get(&context, {i64Type, i32Type, builder.getType<mlir::LLVM::LLVMPointerType>()}, {i64Type});
    auto getTextFunc = builder.create<mlir::func::FuncOp>(location, "get_text_field", funcType);
    getTextFunc.setPrivate();

    funcType = mlir::FunctionType::get(&context, {i64Type}, {i1Type});
    auto addTupleFunc = builder.create<mlir::func::FuncOp>(location, "add_tuple_to_result", funcType);
    addTupleFunc.setPrivate();

    // Create main function that demonstrates typed field access with actual columns
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
    builder.createBlock(&beforeRegion, beforeRegion.end(), argTypes, {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&beforeRegion.front());

    mlir::Value continueFlag = beforeRegion.front().getArgument(0);
    mlir::Value currentCount = beforeRegion.front().getArgument(1);

    builder.create<mlir::scf::ConditionOp>(location, continueFlag, beforeRegion.front().getArguments());

    // After region: read tuple and access fields
    auto& afterRegion = whileOp.getAfter();
    builder.createBlock(&afterRegion, afterRegion.end(), argTypes, {builder.getUnknownLoc(), builder.getUnknownLoc()});
    builder.setInsertionPointToStart(&afterRegion.front());

    mlir::Value tupleCount = afterRegion.front().getArgument(1);

    // Read tuple using pg dialect
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context);
    mlir::OperationState readState(location, mlir::pg::ReadTupleOp::getOperationName());
    readState.addOperands(tableHandle);
    readState.addTypes(tupleHandleType);
    auto readOp = builder.create(readState);
    auto tupleHandle = readOp->getResult(0);

    // Convert tuple to i64 for comparison
    auto tupleAsI64 = builder.create<mlir::UnrealizedConversionCastOp>(location, i64Type, tupleHandle).getResult(0);
    auto isEndOfTable =
        builder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, tupleAsI64, negTwoConst);

    auto ifOp =
        builder.create<mlir::scf::IfOp>(location, llvm::ArrayRef<mlir::Type>{i1Type, i64Type}, isEndOfTable, true);

    // Then branch: end of table
    auto& thenRegion = ifOp.getThenRegion();
    if (thenRegion.empty()) {
        builder.createBlock(&thenRegion);
    }
    builder.setInsertionPointToStart(&thenRegion.front());
    auto falseConst = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(false));
    llvm::SmallVector<mlir::Value> thenYieldOperands = {falseConst, tupleCount};
    builder.create<mlir::scf::YieldOp>(location, thenYieldOperands);

    // Else branch: process tuple with field access for actual selected columns
    auto& elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
        builder.createBlock(&elseRegion);
    }
    builder.setInsertionPointToStart(&elseRegion.front());
    auto trueContinue = builder.create<mlir::arith::ConstantOp>(location, builder.getBoolAttr(true));

    // Access each selected column by its actual index
    for (int columnIndex : selectedColumns) {
        // Determine field type - for now, assume integer types for simplicity
        // In a full implementation, this would use PostgreSQL column type information
        mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
        getFieldState.addOperands(tupleHandle);
        getFieldState.addAttribute("field_index", builder.getI32IntegerAttr(columnIndex));
        getFieldState.addTypes({i32Type, i1Type}); // value and null flag
        auto getFieldOp = builder.create(getFieldState);
        auto fieldValue = getFieldOp->getResult(0);
        auto fieldNullFlag = getFieldOp->getResult(1);

        // For now, we just access the fields to demonstrate correct indexing
        // The actual tuple output still happens via add_tuple_to_result
    }

    // Output the tuple (this will be fixed later to output only selected columns)
    llvm::SmallVector<mlir::Value> addOperands = {tupleAsI64};
    auto addTupleCall = builder.create<mlir::func::CallOp>(location, addTupleFunc, addOperands);

    auto oneIntConst = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(1));
    auto newCount = builder.create<mlir::arith::AddIOp>(location, tupleCount, oneIntConst);

    llvm::SmallVector<mlir::Value> elseYieldOperands = {trueContinue, newCount.getResult()};
    builder.create<mlir::scf::YieldOp>(location, elseYieldOperands);

    // Continue after the while loop
    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::scf::YieldOp>(location, ifOp.getResults());

    builder.setInsertionPointAfter(whileOp);

    // Close table
    auto tableHandleAsInt = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(0));
    llvm::SmallVector<mlir::Value> closeOperands = {tableHandleAsInt};
    builder.create<mlir::func::CallOp>(location, closeFunc, closeOperands);

    mlir::Value finalCount = whileOp.getResult(1);
    builder.create<mlir::func::ReturnOp>(location, finalCount);

    // Print the MLIR with typed field access
    logger.notice("Generated MLIR with PostgreSQL typed field access:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR with field access: " + mlirStr);

    // Apply pg-to-scf lowering pass to convert high-level operations to runtime calls
    mlir::PassManager pm(&context);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::pg::createLowerPgToSCFPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to apply pg-to-scf lowering pass");
        return false;
    }

    logger.notice("Applied pg-to-scf lowering pass!");
    std::string loweredStr;
    llvm::raw_string_ostream loweredOs(loweredStr);
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
    mlir::ExecutionEngineOptions engineOptions;
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
        std::string errMsg;
        llvm::raw_string_ostream errStream(errMsg);
        errStream << expectedFPtr.takeError();
        logger.error("Failed to lookup function: " + errMsg);
        return false;
    }

    auto fptr = reinterpret_cast<int64_t (*)()>(*expectedFPtr);
    int64_t result = fptr();
    logger.notice("Invoked MLIR JIT PostgreSQL typed field access!");

    oss.str("");
    oss << "PostgreSQL typed field access completed with result: " << result;
    logger.notice(oss.str());

    logger.notice("MLIR successfully handled the query");

    return true;
}

} // namespace mlir_runner