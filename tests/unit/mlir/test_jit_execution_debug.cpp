#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"
#include "execution/jit_execution_engine.h"
#include <sstream>

namespace {

class JITExecutionDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create context with all dialects
        context = std::make_shared<mlir::MLIRContext>();
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                       mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                       mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                       mlir::relalg::RelAlgDialect, mlir::db::DBDialect,
                       mlir::dsa::DSADialect, mlir::util::UtilDialect>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
    }
    
    std::shared_ptr<mlir::MLIRContext> context;
};

// Test 1: Verify "main" function survives Standard→LLVM conversion
TEST_F(JITExecutionDebugTest, TestMainFunctionPreservationThroughLLVM) {
    PGX_INFO("=== Testing 'main' function preservation through Standard→LLVM ===");
    
    // Create module with "main" function (like PostgreSQL creates)
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create "main" function returning i32 (like PostgreSQL)
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add simple operation + return
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    // Verify "main" exists before lowering
    auto mainFuncBefore = module.lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_NE(mainFuncBefore, nullptr) << "main function should exist before lowering";
    PGX_INFO("✓ 'main' function exists before Standard→LLVM lowering");
    
    // Apply Standard→LLVM pipeline (same as PostgreSQL)
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result)) << "Standard→LLVM lowering should succeed";
    
    // Verify "main" exists after lowering (CRITICAL TEST)
    auto mainFuncAfter = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (mainFuncAfter) {
        PGX_INFO("✓ 'main' function PRESERVED through Standard→LLVM lowering");
    } else {
        PGX_ERROR("✗ 'main' function LOST during Standard→LLVM lowering!");
        
        // Debug: Print all functions in module
        PGX_INFO("Functions remaining after Standard→LLVM:");
        module->walk([&](mlir::func::FuncOp funcOp) {
            PGX_INFO("  - Function: " + std::string(funcOp.getName()));
        });
        
        FAIL() << "Main function disappeared during Standard→LLVM lowering";
    }
    
    // Print final module for debugging
    PGX_INFO("Final LLVM module:");
    std::string moduleStr;
    llvm::raw_string_ostream os(moduleStr);
    module->print(os, mlir::OpPrintingFlags().assumeVerified());
    os.flush();
    
    std::istringstream iss(moduleStr);
    std::string line;
    while (std::getline(iss, line)) {
        PGX_INFO("LLVM: " + line);
    }
}

// Test 2: Test full pipeline + JIT execution (isolate JIT issue)
TEST_F(JITExecutionDebugTest, TestCompleteStandardToLLVMToJIT) {
    PGX_INFO("=== Testing complete Standard→LLVM→JIT pipeline ===");
    
    // Create module with "main" function
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create "main" function returning i32
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    // Apply Standard→LLVM pipeline
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    auto loweringResult = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(loweringResult)) << "Standard→LLVM lowering must succeed";
    
    // Test JIT execution (isolate the exact failure point)
    auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
    
    // Check if JIT initialization succeeds
    bool jitInitSuccess = jitEngine->initialize();
    if (!jitInitSuccess) {
        PGX_ERROR("✗ JIT initialization failed");
        FAIL() << "JIT engine initialization failed";
    }
    PGX_INFO("✓ JIT engine initialized successfully");
    
    // Test function lookup (this is where PostgreSQL fails)
    bool lookupSuccess = jitEngine->lookupCompiledQuery();
    if (!lookupSuccess) {
        PGX_ERROR("✗ JIT function lookup failed - this is the PostgreSQL bug!");
        
        // Debug: Check what functions are available
        PGX_INFO("Module contains these functions:");
        module->walk([&](mlir::func::FuncOp funcOp) {
            PGX_INFO("  - Available function: " + std::string(funcOp.getName()));
        });
        
        // Check LLVM module directly
        PGX_INFO("Raw LLVM module content:");
        module->dump();
        
        FAIL() << "JIT function lookup failed - matches PostgreSQL error";
    }
    
    PGX_INFO("✓ JIT function lookup succeeded!");
    
    // If we get here, JIT execution should work
    // Test actual execution (if lookup succeeds)
    PGX_INFO("✓ Complete Standard→LLVM→JIT pipeline works!");
}

// Test 3: Test "query_main" vs "main" naming issue
TEST_F(JITExecutionDebugTest, TestFunctionNamingConsistency) {
    PGX_INFO("=== Testing function naming consistency ===");
    
    // Test both naming patterns that appear in our codebase
    std::vector<std::string> functionNames = {"main", "query_main"};
    
    for (const auto& funcName : functionNames) {
        PGX_INFO("Testing function name: " + funcName);
        
        // Create module
        mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
        mlir::OpBuilder builder(module.getBodyRegion());
        
        // Create function with test name
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), funcName, funcType);
        
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto constant = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 42, 32);
        builder.create<mlir::func::ReturnOp>(
            builder.getUnknownLoc(), constant.getResult());
        
        // Apply Standard→LLVM
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        auto result = pm.run(module);
        EXPECT_TRUE(mlir::succeeded(result));
        
        // Check if function survives with same name
        auto funcAfter = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        if (funcAfter) {
            PGX_INFO("✓ Function '" + funcName + "' preserved through lowering");
        } else {
            PGX_ERROR("✗ Function '" + funcName + "' lost during lowering");
            
            // See what names exist instead
            module->walk([&](mlir::func::FuncOp f) {
                PGX_INFO("  Found function: " + std::string(f.getName()));
            });
        }
        
        EXPECT_NE(funcAfter, nullptr) << "Function " << funcName << " should survive lowering";
    }
}

// Test 4: Replicate exact PostgreSQL pipeline in unit test
TEST_F(JITExecutionDebugTest, TestPostgreSQLExactPipelineReplication) {
    PGX_INFO("=== Replicating exact PostgreSQL pipeline sequence ===");
    
    // Initialize UtilDialect like PostgreSQL does
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    auto* utilDialect = context->getOrLoadDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr);
    utilDialect->getFunctionHelper().setParentModule(module);
    
    // Create RelAlg operations like PostgreSQL AST translator does
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create "main" function (like postgresql_ast_translator.cpp:195-198)
    auto queryFuncType = builder.getFunctionType({}, {builder.getI32Type()});
    auto queryFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", queryFuncType);
    
    auto* entryBlock = queryFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add placeholder operations that get generated by RelAlg
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    PGX_INFO("Initial module created with 'main' function");
    
    // Phase 1: Skip RelAlg creation (already done)
    // Phase 2: Skip RelAlg→DB (we're testing Standard→LLVM→JIT)
    // Phase 3: Standard→LLVM (this is where function disappears)
    {
        PGX_INFO("Running Phase 3c: Standard→LLVM lowering");
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        
        auto result = pm.run(module);
        ASSERT_TRUE(mlir::succeeded(result)) << "Standard→LLVM must succeed";
        PGX_INFO("Phase 3c completed: Standard MLIR successfully lowered to LLVM IR");
    }
    
    // Phase 4: JIT Execution (this is where PostgreSQL fails)
    {
        PGX_INFO("Phase 4g-2: JIT Execution");
        
        // Check if main function exists before JIT
        auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
        if (mainFunc) {
            PGX_INFO("✓ 'main' function exists before JIT execution");
        } else {
            PGX_ERROR("✗ 'main' function missing before JIT - this explains PostgreSQL failure!");
            FAIL() << "Function disappeared between Standard→LLVM lowering and JIT";
        }
        
        // Try JIT execution like PostgreSQL does
        auto jitEngine = std::make_unique<::pgx_lower::JITExecutionEngine>(module);
        bool initSuccess = jitEngine->initialize();
        EXPECT_TRUE(initSuccess) << "JIT initialization should succeed";
        
        if (initSuccess) {
            bool lookupSuccess = jitEngine->lookupCompiledQuery();
            if (lookupSuccess) {
                PGX_INFO("✓ JIT execution setup succeeded - PostgreSQL bug is NOT in function naming");
            } else {
                PGX_ERROR("✗ JIT function lookup failed - this IS the PostgreSQL bug");
                PGX_INFO("This unit test has isolated the exact failure point in PostgreSQL tests");
            }
            EXPECT_TRUE(lookupSuccess) << "JIT function lookup should succeed";
        }
    }
}

} // namespace