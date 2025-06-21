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
    mlir::PassPipelineRegistration<>(
        "convert-to-llvm", "Convert MLIR to LLVM dialect",
        [](mlir::OpPassManager& pm) {
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createConvertSCFToCFPass());
        });
}

auto run_mlir_core(int64_t intValue, MLIRLogger& logger) -> bool {
    // Create MLIR context and builder
    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "MLIRContext symbol address: " << (void*)&context;
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
    mlir::Block* entryBlock = func.addEntryBlock();
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
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to lower MLIR module to LLVM dialect");
        return false;
    }
    logger.notice("Lowered MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    // Register all translation interfaces directly with the context
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context.appendDialectRegistry(registry);
    
    // Also register individual translations to ensure completeness
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

    auto fptr = reinterpret_cast<int64_t (*)()>(*expectedFPtr);
    int64_t result = fptr();
    logger.notice("Invoked MLIR JIT-compiled function!");

    oss.str("");
    oss << "MLIR JIT returned: " << result;
    logger.notice(oss.str());

    return true;
}

static ExternalFunction* g_external_func = nullptr;

extern "C" int64_t call_external_function() {
    if (g_external_func && *g_external_func) {
        return (*g_external_func)();
    }
    return -1;
}

bool run_mlir_with_external_func(int64_t intValue,
                                 const ExternalFunction& externalFunc,
                                 MLIRLogger& logger) {
    g_external_func = const_cast<ExternalFunction*>(&externalFunc);

    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "MLIRContext symbol address: " << (void*)&context;
    logger.debug(oss.str());

    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    registerConversionPipeline();

    std::unique_ptr<mlir::ExecutionEngine> engine;
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    mlir::Location loc = builder.getUnknownLoc();

    auto externalFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto externalFuncDecl = mlir::func::FuncOp::create(
        loc, "call_external_function", externalFuncType);
    externalFuncDecl.setPrivate();
    module.push_back(externalFuncDecl);

    auto mainFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto mainFunc = mlir::func::FuncOp::create(loc, "main", mainFuncType);
    mainFunc.setPublic();
    mlir::Block* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto callOp = builder.create<mlir::func::CallOp>(loc, externalFuncDecl,
                                                     mlir::ValueRange{});

    auto constOp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(intValue));
    auto addOp = builder.create<mlir::arith::AddIOp>(loc, callOp.getResult(0),
                                                     constOp.getResult());

    builder.create<mlir::func::ReturnOp>(loc, addOp.getResult());
    module.push_back(mainFunc);

    if (mlir::failed(mlir::verify(module))) {
        logger.error("MLIR module verification failed");
        return false;
    }

    logger.notice("Generated MLIR program with external function call:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR: " + mlirStr);

    mlir::PassManager pm(&context);
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to lower MLIR module to LLVM dialect");
        return false;
    }
    logger.notice("Lowered MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    // Register all translation interfaces directly with the context
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context.appendDialectRegistry(registry);
    
    // Also register individual translations to ensure completeness
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

    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        symbolMap[interner("call_external_function")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void*>(&call_external_function)),
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
    logger.notice("Invoked MLIR JIT-compiled function with external call!");

    oss.str("");
    oss << "MLIR JIT returned: " << result;
    logger.notice(oss.str());

    g_external_func = nullptr;

    return true;
}

bool run_mlir_with_multi_tuple_scan(const ExternalFunction& externalFunc,
                                    MLIRLogger& logger) {
    g_external_func = const_cast<ExternalFunction*>(&externalFunc);

    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "MLIRContext symbol address: " << (void*)&context;
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

    auto externalFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto externalFuncDecl = mlir::func::FuncOp::create(
        loc, "call_external_function", externalFuncType);
    externalFuncDecl.setPrivate();
    module.push_back(externalFuncDecl);

    auto mainFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto mainFunc = mlir::func::FuncOp::create(loc, "main", mainFuncType);
    mainFunc.setPublic();
    mlir::Block* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto zeroConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(0));
    auto negTwoConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(-2));

    mlir::Value sum = zeroConst;

    for (int i = 0; i < 10; i++) {
        auto callOp = builder.create<mlir::func::CallOp>(loc, externalFuncDecl,
                                                         mlir::ValueRange{});
        mlir::Value tupleValue = callOp.getResult(0);

        auto cmpOp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, tupleValue, negTwoConst);

        auto ifOp = builder.create<mlir::scf::IfOp>(loc, builder.getI64Type(),
                                                    cmpOp, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        builder.create<mlir::scf::YieldOp>(loc, sum);

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        auto newSum = builder.create<mlir::arith::AddIOp>(loc, sum, tupleValue);
        builder.create<mlir::scf::YieldOp>(loc, newSum.getResult());

        builder.setInsertionPointAfter(ifOp);
        sum = ifOp.getResult(0);
    }

    builder.create<mlir::func::ReturnOp>(loc, sum);

    module.push_back(mainFunc);

    if (mlir::failed(mlir::verify(module))) {
        logger.error("MLIR module verification failed");
        return false;
    }

    logger.notice("Generated MLIR program with multi-tuple scan:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR: " + mlirStr);

    mlir::PassManager pm(&context);
    // First convert SCF to control flow
    pm.addPass(mlir::createConvertSCFToCFPass());
    // Then convert all high-level dialects to LLVM
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    // Add control flow to LLVM pass 
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to lower MLIR module to LLVM dialect");
        return false;
    }
    logger.notice("Lowered MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    // Register all translation interfaces directly with the context
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context.appendDialectRegistry(registry);
    
    // Also register individual translations to ensure completeness
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

    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        symbolMap[interner("call_external_function")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void*>(&call_external_function)),
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
    logger.notice("Invoked MLIR JIT-compiled multi-tuple function!");

    oss.str("");
    oss << "MLIR JIT returned sum: " << result;
    logger.notice(oss.str());

    g_external_func = nullptr;

    return true;
}

struct DataAccessContext {
    void* dataPtr;
    size_t size;
    size_t chunkSize;
};

static DataAccessContext* g_data_context = nullptr;

extern "C" int64_t read_int64_from_memory(void* ptr, size_t offset) {
    char* data = static_cast<char*>(ptr);
    int64_t value;
    memcpy(&value, data + offset, sizeof(int64_t));
    return value;
}

extern "C" int64_t get_data_size() {
    return g_data_context ? g_data_context->size : 0;
}

// PostgreSQL table access functions - these will be called from MLIR JIT
extern "C" void* open_postgres_table(const char* tableName);
extern "C" int64_t read_next_tuple_from_table(void* tableHandle);
extern "C" void close_postgres_table(void* tableHandle);

bool run_mlir_with_direct_data_access(void* dataPtr, size_t dataSize,
                                      MLIRLogger& logger) {
    static DataAccessContext dataContext;
    dataContext = {dataPtr, dataSize, sizeof(int64_t)};
    g_data_context = &dataContext;

    size_t numElements = dataSize / sizeof(int64_t);
    if (numElements == 0) {
        logger.error("Invalid data size: no int64_t elements found");
        g_data_context = nullptr;
        return false;
    }

    mlir::MLIRContext context;

    std::ostringstream oss;
    oss << "Processing " << numElements << " int64_t elements (" << dataSize << " bytes) directly in MLIR";
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

    auto memReadFuncType = builder.getFunctionType(
        {builder.getI64Type(), builder.getI64Type()}, {builder.getI64Type()});
    auto memReadFunc = mlir::func::FuncOp::create(loc, "read_int64_from_memory",
                                                  memReadFuncType);
    memReadFunc.setPrivate();
    module.push_back(memReadFunc);

    auto dataSizeFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto dataSizeFunc =
        mlir::func::FuncOp::create(loc, "get_data_size", dataSizeFuncType);
    dataSizeFunc.setPrivate();
    module.push_back(dataSizeFunc);

    auto mainFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto mainFunc = mlir::func::FuncOp::create(loc, "main", mainFuncType);
    mainFunc.setPublic();
    mlir::Block* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto zeroConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(0));
    auto eightConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(8));
    auto ptrConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(reinterpret_cast<int64_t>(dataPtr)));

    mlir::Value sum = zeroConst;

    // Use SCF for loop for variable-size data processing 
    auto lowerBound = zeroConst;
    auto upperBound = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(static_cast<int64_t>(numElements)));
    auto step = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(1));

    auto forOp = builder.create<mlir::scf::ForOp>(
        loc, lowerBound, upperBound, step, mlir::ValueRange{sum});

    // Inside the for loop
    builder.setInsertionPointToStart(forOp.getBody());
    mlir::Value index = forOp.getInductionVar();
    mlir::Value currentSum = forOp.getRegionIterArg(0);

    // Calculate byte offset: index * 8
    auto byteOffset = builder.create<mlir::arith::MulIOp>(loc, index, eightConst);

    auto readCall = builder.create<mlir::func::CallOp>(
        loc, memReadFunc, mlir::ValueRange{ptrConst, byteOffset});
    mlir::Value value = readCall.getResult(0);

    auto newSum = builder.create<mlir::arith::AddIOp>(loc, currentSum, value);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newSum.getResult()});

    // After the for loop
    builder.setInsertionPointAfter(forOp);
    sum = forOp.getResult(0);

    builder.create<mlir::func::ReturnOp>(loc, sum);

    module.push_back(mainFunc);

    if (mlir::failed(mlir::verify(module))) {
        logger.error("MLIR module verification failed");
        g_data_context = nullptr;
        return false;
    }

    logger.notice("Generated MLIR program with direct memory access:");
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.OpState::print(os);
    os.flush();
    logger.notice("MLIR: " + mlirStr);

    mlir::PassManager pm(&context);
    // First convert SCF to control flow
    pm.addPass(mlir::createConvertSCFToCFPass());
    // Then convert all high-level dialects to LLVM
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    // Add control flow to LLVM pass 
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
        logger.error("Failed to lower MLIR module to LLVM dialect");
        g_data_context = nullptr;
        return false;
    }
    logger.notice("Lowered MLIR to LLVM dialect!");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    // Register all translation interfaces directly with the context
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    context.appendDialectRegistry(registry);
    
    // Also register individual translations to ensure completeness
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);

    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        logger.error("Failed to create MLIR ExecutionEngine");
        g_data_context = nullptr;
        return false;
    }
    logger.notice("Created MLIR ExecutionEngine for direct data access!");
    engine = std::move(*maybeEngine);

    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        symbolMap[interner("read_int64_from_memory")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void*>(&read_int64_from_memory)),
                llvm::JITSymbolFlags::Exported);
        symbolMap[interner("get_data_size")] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(
                reinterpret_cast<void*>(&get_data_size)),
            llvm::JITSymbolFlags::Exported);
        return symbolMap;
    });

    auto expectedFPtr = engine->lookup("main");
    if (!expectedFPtr) {
        std::string errMsg;
        llvm::raw_string_ostream errStream(errMsg);
        errStream << expectedFPtr.takeError();
        logger.error("Failed to lookup function: " + errMsg);
        g_data_context = nullptr;
        return false;
    }

    auto fptr = reinterpret_cast<int64_t (*)()>(*expectedFPtr);
    int64_t result = fptr();
    logger.notice("Invoked MLIR JIT with direct memory access!");

    oss.str("");
    oss << "MLIR processed " << dataSize
        << " bytes and returned sum: " << result;
    logger.notice(oss.str());

    g_data_context = nullptr;

    return true;
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

    auto mainFuncType = builder.getFunctionType({}, {builder.getI64Type()});
    auto mainFunc = mlir::func::FuncOp::create(loc, "main", mainFuncType);
    mainFunc.setPublic();
    mlir::Block* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto zeroConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(0));
    auto negTwoConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(-2)); // End of table marker
    auto tableNamePtr = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(reinterpret_cast<int64_t>(tableName)));

    auto openCall = builder.create<mlir::func::CallOp>(
        loc, openTableFunc, mlir::ValueRange{tableNamePtr});
    mlir::Value tableHandle = openCall.getResult(0);

    auto maxRowsConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(10)); // Process max 10 rows for testing
    auto oneConst = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(1));

    auto forOp = builder.create<mlir::scf::ForOp>(
        loc, zeroConst, maxRowsConst, oneConst, mlir::ValueRange{zeroConst});

    auto* loopBody = forOp.getBody();
    builder.setInsertionPointToStart(loopBody);
    
    mlir::Value loopSum = forOp.getRegionIterArgs()[0];
    
    auto readCall = builder.create<mlir::func::CallOp>(
        loc, readTupleFunc, mlir::ValueRange{tableHandle});
    mlir::Value tupleValue = readCall.getResult(0);

    auto isEndOfTable = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, tupleValue, negTwoConst);
    
    auto resultSum = builder.create<mlir::scf::IfOp>(
        loc, builder.getI64Type(), isEndOfTable, true);
    
    auto& thenRegion = resultSum.getThenRegion();
    if (thenRegion.empty()) {
        builder.createBlock(&thenRegion);
    }
    builder.setInsertionPointToStart(&thenRegion.front());
    builder.create<mlir::scf::YieldOp>(loc, loopSum);
    
    auto& elseRegion = resultSum.getElseRegion();
    if (elseRegion.empty()) {
        builder.createBlock(&elseRegion);
    }
    builder.setInsertionPointToStart(&elseRegion.front());
    auto newSum = builder.create<mlir::arith::AddIOp>(loc, loopSum, tupleValue);
    builder.create<mlir::scf::YieldOp>(loc, newSum.getResult());
    
    builder.setInsertionPointAfter(resultSum);
    builder.create<mlir::scf::YieldOp>(loc, resultSum.getResult(0));
    
    builder.setInsertionPointAfter(forOp);
    mlir::Value finalSum = forOp.getResult(0);

    builder.create<mlir::func::CallOp>(loc, closeTableFunc, mlir::ValueRange{tableHandle});

    builder.create<mlir::func::ReturnOp>(loc, finalSum);

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
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
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
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void*>(&open_postgres_table)),
                llvm::JITSymbolFlags::Exported);
        symbolMap[interner("read_next_tuple_from_table")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void*>(&read_next_tuple_from_table)),
                llvm::JITSymbolFlags::Exported);
        symbolMap[interner("close_postgres_table")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void*>(&close_postgres_table)),
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

#ifndef POSTGRESQL_EXTENSION
bool run_mlir_test(int64_t intValue) {
    ConsoleLogger logger;
    return run_mlir_core(intValue, logger);
}

bool run_external_func_test(const ExternalFunction& externalFunc) {
    ConsoleLogger logger;
    return run_mlir_with_external_func(10, externalFunc, logger);
}
#endif

}  // namespace mlir_runner