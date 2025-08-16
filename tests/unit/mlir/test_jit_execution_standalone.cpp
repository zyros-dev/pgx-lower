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
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "execution/logging.h"
#include "llvm/Support/TargetSelect.h"

// Include all our conversion passes
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "pgx_lower/mlir/Conversion/UtilToLLVM/Passes.h"

// Test result tracking
static int g_test_execution_counter = 0;
static bool g_test_function_executed = false;

// Runtime function stubs for JIT execution
extern "C" {

// Test stub that proves function execution
void test_execution_marker() {
    g_test_execution_counter++;
    g_test_function_executed = true;
    std::cout << "🎯 JIT FUNCTION EXECUTED! Counter: " << g_test_execution_counter << std::endl;
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
    PGX_INFO("🧪 TEST: Executing minimal function with ExecutionEngine");
    
    registerAllDialects();
    
    // Create the simplest possible function that calls our test marker
    builder.setInsertionPointToEnd(module.getBody());
    
    // Declare the test marker function
    auto testMarkerFuncType = builder.getFunctionType({}, {});
    auto testMarkerFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), 
        "test_execution_marker", 
        testMarkerFuncType);
    testMarkerFunc.setPrivate(); // External function
    
    // Create main function
    auto mainFuncType = builder.getFunctionType({}, {});
    auto mainFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", mainFuncType);
    
    auto* block = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Call the test marker to prove execution
    auto testMarkerSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), "test_execution_marker");
    builder.create<mlir::func::CallOp>(
        builder.getUnknownLoc(),
        testMarkerSymbol,
        mlir::TypeRange{}, // void return type
        mlir::ValueRange{});
    
    // Add simple arithmetic to test basic operations
    auto constOne = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(),
        builder.getI32IntegerAttr(1));
    
    auto constTwo = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(),
        builder.getI32IntegerAttr(2));
    
    auto addResult = builder.create<mlir::arith::AddIOp>(
        builder.getUnknownLoc(),
        constOne,
        constTwo);
    
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    PGX_INFO("📋 Created function with test marker call and arithmetic");
    
    // Lower to LLVM IR
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module))) << "Standard→LLVM lowering failed";
    PGX_INFO("✅ Standard→LLVM lowering succeeded");
    
    // Create ExecutionEngine
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        PGX_ERROR("ExecutionEngine creation failed: " + llvm::toString(maybeEngine.takeError()));
        ASSERT_TRUE(false) << "ExecutionEngine creation failed";
    }
    
    auto engine = std::move(*maybeEngine);
    PGX_INFO("✅ ExecutionEngine created successfully");
    
    // Register our test runtime function
    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        
        symbolMap[interner("test_execution_marker")] = {
            llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(test_execution_marker)),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable
        };
        
        return symbolMap;
    });
    
    PGX_INFO("🔧 Registered test execution marker function");
    
    // Verify function lookup works
    auto lookupResult = engine->lookup("main");
    EXPECT_TRUE(static_cast<bool>(lookupResult)) << "Main function lookup should succeed";
    
    auto testMarkerLookup = engine->lookup("test_execution_marker");
    EXPECT_TRUE(static_cast<bool>(testMarkerLookup)) << "Test marker function lookup should succeed";
    
    if (lookupResult && testMarkerLookup) {
        PGX_INFO("✅ Both functions found in ExecutionEngine");
    }
    
    // Execute using invoke() method
    PGX_INFO("🎯 CRITICAL TEST: Calling engine->invoke('main')");
    PGX_INFO("Pre-execution: counter=" + std::to_string(g_test_execution_counter) + 
             ", executed=" + std::to_string(g_test_function_executed));
    
    auto invokeResult = engine->invoke("main");
    
    PGX_INFO("Post-execution: counter=" + std::to_string(g_test_execution_counter) + 
             ", executed=" + std::to_string(g_test_function_executed));
    PGX_INFO("Invoke result: " + std::to_string(static_cast<bool>(invokeResult)));
    
    // Test results
    EXPECT_TRUE(static_cast<bool>(invokeResult)) << "invoke('main') should succeed";
    EXPECT_TRUE(g_test_function_executed) << "Test marker function should have been called";
    EXPECT_GT(g_test_execution_counter, 0) << "Execution counter should be incremented";
    
    if (g_test_function_executed) {
        PGX_INFO("🎉 SUCCESS: JIT function execution works in unit test environment!");
        PGX_INFO("This proves ExecutionEngine can execute functions when properly configured");
    } else {
        PGX_ERROR("❌ FAILURE: JIT function execution failed even in unit test environment");
        PGX_ERROR("This suggests a fundamental issue with our ExecutionEngine usage");
    }
}

TEST_F(JITExecutionStandaloneTest, ExecuteWithRuntimeFunctions) {
    PGX_INFO("🧪 TEST: Execute LLVM IR with external runtime function calls");
    
    registerAllDialects();
    
    // Create LLVM IR module that calls external runtime functions (like our real case)
    builder.setInsertionPointToEnd(module.getBody());
    
    // Declare external runtime functions (the key ones from our real module)
    auto voidType = builder.getNoneType();
    auto refType = mlir::LLVM::LLVMPointerType::get(&context);
    auto i32Type = builder.getI32Type();
    
    // Declare test marker
    auto testMarkerFuncType = mlir::LLVM::LLVMFunctionType::get(voidType, {});
    auto testMarkerFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "test_execution_marker", testMarkerFuncType);
    testMarkerFunc.setSymVisibilityAttr(builder.getStringAttr("private"));
    
    // Declare some key runtime functions that our real module calls
    auto contextFuncType = mlir::LLVM::LLVMFunctionType::get(refType, {});
    auto contextFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "rt_get_execution_context", contextFuncType);
    contextFunc.setSymVisibilityAttr(builder.getStringAttr("private"));
    
    auto tableCreateFuncType = mlir::LLVM::LLVMFunctionType::get(refType, {refType});
    auto tableCreateFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "_ZN7runtime12TableBuilder6createENS_8VarLen32E", tableCreateFuncType);
    tableCreateFunc.setSymVisibilityAttr(builder.getStringAttr("private"));
    
    // Create main function that calls external functions (like our real case)
    auto mainFuncType = mlir::LLVM::LLVMFunctionType::get(voidType, {});
    auto mainFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "main", mainFuncType);
    mainFunc.setSymVisibilityAttr(builder.getStringAttr("public"));
    
    auto* block = mainFunc.addEntryBlock(builder);
    builder.setInsertionPointToEnd(block);
    
    // Call test marker to prove execution
    builder.create<mlir::LLVM::CallOp>(
        builder.getUnknownLoc(),
        testMarkerFunc,
        mlir::ValueRange{});
    
    // Call runtime functions (the key test - can JIT call external functions?)
    builder.create<mlir::LLVM::CallOp>(
        builder.getUnknownLoc(),
        contextFunc,
        mlir::ValueRange{});
    
    // Create null pointer for TableBuilder::create call
    auto nullPtr = builder.create<mlir::LLVM::ZeroOp>(builder.getUnknownLoc(), refType);
    builder.create<mlir::LLVM::CallOp>(
        builder.getUnknownLoc(),
        tableCreateFunc,
        mlir::ValueRange{nullPtr});
    
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    
    PGX_INFO("📋 Created LLVM module with external runtime function calls");
    
    // Create ExecutionEngine and test execution
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        PGX_ERROR("ExecutionEngine creation failed: " + llvm::toString(maybeEngine.takeError()));
        ASSERT_TRUE(false) << "ExecutionEngine creation failed";
    }
    
    auto engine = std::move(*maybeEngine);
    
    // Register the external runtime functions
    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        
        // Test marker
        symbolMap[interner("test_execution_marker")] = 
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(test_execution_marker)),
                                        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
        
        // Key runtime functions from runtime_stubs.cpp
        symbolMap[interner("rt_get_execution_context")] = 
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(rt_get_execution_context)),
                                        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
        
        symbolMap[interner("_ZN7runtime12TableBuilder6createENS_8VarLen32E")] = 
            llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(_ZN7runtime12TableBuilder6createENS_8VarLen32E)),
                                        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
        
        return symbolMap;
    });
    
    PGX_INFO("🔧 Registered runtime functions");
    
    // Execute with external function calls
    PGX_INFO("🎯 CRITICAL TEST: JIT execution with external runtime function calls");
    PGX_INFO("Pre-execution: counter=" + std::to_string(g_test_execution_counter) + 
             ", executed=" + std::to_string(g_test_function_executed));
    
    auto invokeResult = engine->invoke("main");
    
    PGX_INFO("Post-execution: counter=" + std::to_string(g_test_execution_counter) + 
             ", executed=" + std::to_string(g_test_function_executed));
    PGX_INFO("Invoke result: " + std::to_string(static_cast<bool>(invokeResult)));
    
    // Test results
    EXPECT_TRUE(static_cast<bool>(invokeResult)) << "JIT execution with runtime calls should succeed";
    EXPECT_TRUE(g_test_function_executed) << "Test marker should prove function executed";
    EXPECT_GT(g_test_execution_counter, 0) << "Execution counter should be incremented";
    
    if (g_test_function_executed) {
        PGX_INFO("🎉 SUCCESS: JIT execution with external runtime functions works!");
        PGX_INFO("This proves ExecutionEngine can call external functions properly");
    } else {
        PGX_ERROR("❌ CRITICAL: JIT execution with runtime calls failed");
        PGX_ERROR("This suggests ExecutionEngine cannot resolve external function calls");
    }
}