#include <gtest/gtest.h>
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "execution/logging.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"  // For DynamicLibrarySearchGenerator
#include "mlir/Parser/Parser.h"
#include <fstream>
#include <dlfcn.h>  // For dlopen, dlsym, dlclose
#include <cstdlib>  // For system() calls

// Include all our conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "pgx_lower/mlir/Conversion/UtilToLLVM/Passes.h"

// Test result tracking
static int g_test_execution_counter = 0;
static bool g_test_function_executed = false;

// Runtime function stubs for JIT execution
// CRITICAL: extern "C" prevents C++ name mangling and ensures C calling convention
extern "C" {

// Test stub that proves function execution - MUST match LLVM IR signature exactly
// DEEPWIKI SUGGESTION: extern "C" prevents C++ name mangling to match llvm.func @test_execution_marker()
void test_execution_marker() {
    // CRITICAL: Use stderr to bypass any buffering issues
    fprintf(stderr, "🎯 JIT FUNCTION EXECUTED! About to increment counter\n");
    fflush(stderr);
    
    g_test_execution_counter++;
    g_test_function_executed = true;
    
    fprintf(stderr, "🎯 JIT FUNCTION: Counter now = %d, executed = %s\n", 
            g_test_execution_counter, g_test_function_executed ? "true" : "false");
    fflush(stderr);
    
    std::cout << "🎯 JIT FUNCTION EXECUTED! Counter: " << g_test_execution_counter << std::endl;
    fflush(stdout);
}

// Declare runtime stubs here so they're available for symbol registration
void* rt_get_execution_context() { return nullptr; }
void* _ZN7runtime12TableBuilder6createENS_8VarLen32E(void* varlen) { return nullptr; }

} // extern "C"

class JITExecutionStandaloneTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    JITExecutionStandaloneTest() : builder(&context) {
        // Initialize LLVM targets for JIT compilation
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        
        // Load all required dialects
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::memref::MemRefDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
        context.loadDialect<mlir::relalg::RelAlgDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        
        // Create module
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        // Reset test state
        g_test_execution_counter = 0;
        g_test_function_executed = false;
    }
    
    void registerAllDialects() {
        mlir::DialectRegistry registry;
        mlir::registerAllToLLVMIRTranslations(registry);
        context.appendDialectRegistry(registry);
        mlir::registerLLVMDialectTranslation(context);
    }
};

TEST_F(JITExecutionStandaloneTest, ExecuteMinimalFunction) {
    // Register LLVM dialect translation FIRST (from working version!)
    registerAllDialects();
    
    // Reset test state
    g_test_execution_counter = 0;
    g_test_function_executed = false;
    
    // WORKING VERSION APPROACH: Use func dialect with llvm.emit_c_interface!
    builder.setInsertionPointToEnd(module.getBody());
    
    // External function declaration using func dialect (working July 30 approach!)
    auto voidType = builder.getNoneType();
    auto testMarkerFuncType = mlir::FunctionType::get(&context, {}, {});
    auto testMarkerFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_execution_marker", testMarkerFuncType);
    testMarkerFunc->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(&context));
    
    // Main function using func dialect
    auto mainFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", testMarkerFuncType);
    mainFunc->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(&context));
    
    auto* block = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), testMarkerFunc, mlir::ValueRange{});
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    
    // Need to lower func dialect to LLVM first (working version did this!)
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    if (mlir::failed(pm.run(module))) {
        FAIL() << "Failed to lower func dialect to LLVM";
    }
    
    // Create ExecutionEngine WITH OPTIMIZER (from working July 30 version!)
    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;  // CRITICAL: This was in working version!
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    ASSERT_TRUE(static_cast<bool>(maybeEngine)) << "ExecutionEngine creation failed";
    auto engine = std::move(*maybeEngine);
    
    // Register the external function (both with and without C interface wrapper)
    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        // Register both the raw function and the C interface wrapper
        symbolMap[interner("test_execution_marker")] = {
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(test_execution_marker)),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
        };
        symbolMap[interner("_mlir_ciface_test_execution_marker")] = {
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(test_execution_marker)),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
        };
        return symbolMap;
    });
    
    // Execute (invoke uses _mlir_ciface wrapper)
    auto invokeResult = engine->invokePacked("main");
    EXPECT_FALSE(static_cast<bool>(invokeResult)) << "invoke('main') should succeed";
    
    // Test results
    EXPECT_TRUE(g_test_function_executed) << "External function should have been called";
    EXPECT_GT(g_test_execution_counter, 0) << "Counter should be incremented";
}
