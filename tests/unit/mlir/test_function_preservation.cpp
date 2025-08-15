#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
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
#include "mlir/Passes.h"
#include <sstream>

namespace {

class FunctionPreservationTest : public ::testing::Test {
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

// Test the exact issue from PostgreSQL: Does "main" function survive Standard→LLVM?
TEST_F(FunctionPreservationTest, MainFunctionSurvivesStandardToLLVM) {
    PGX_INFO("=== CRITICAL TEST: Does 'main' function survive Standard→LLVM lowering? ===");
    
    // Create module with "main" function (exactly like PostgreSQL does)
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create "main" function with i32 return (matches postgresql_ast_translator.cpp:195-198)
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add simple content (like PostgreSQL does)
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    // VERIFY: "main" function exists BEFORE lowering
    auto mainFuncBefore = module.lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_NE(mainFuncBefore, nullptr) << "SETUP ERROR: main function should exist before lowering";
    PGX_INFO("✓ BEFORE: 'main' function exists in module");
    
    // Print module before lowering
    PGX_INFO("Module BEFORE Standard→LLVM:");
    std::string moduleStrBefore;
    llvm::raw_string_ostream osBefore(moduleStrBefore);
    module->print(osBefore, mlir::OpPrintingFlags().assumeVerified());
    osBefore.flush();
    
    std::istringstream issBefore(moduleStrBefore);
    std::string line;
    while (std::getline(issBefore, line)) {
        PGX_INFO("BEFORE: " + line);
    }
    
    // Apply Standard→LLVM lowering (SAME PIPELINE AS POSTGRESQL)
    PGX_INFO("Applying Standard→LLVM lowering pipeline...");
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result)) << "BLOCKER: Standard→LLVM lowering failed";
    PGX_INFO("✓ Standard→LLVM lowering completed successfully");
    
    // CRITICAL TEST: Does "main" function still exist AFTER lowering?
    // After Standard→LLVM lowering, look for llvm.func, not func.func
    auto mainFuncAfter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
    
    // Print module after lowering (for debugging)
    PGX_INFO("Module AFTER Standard→LLVM:");
    std::string moduleStrAfter;
    llvm::raw_string_ostream osAfter(moduleStrAfter);
    module->print(osAfter, mlir::OpPrintingFlags().assumeVerified());
    osAfter.flush();
    
    std::istringstream issAfter(moduleStrAfter);
    while (std::getline(issAfter, line)) {
        PGX_INFO("AFTER: " + line);
    }
    
    // Print all functions in module for debugging
    PGX_INFO("All functions after Standard→LLVM lowering:");
    size_t funcCount = 0;
    module->walk([&](mlir::func::FuncOp funcOp) {
        funcCount++;
        PGX_INFO("  Function " + std::to_string(funcCount) + ": " + std::string(funcOp.getName()));
    });
    
    if (funcCount == 0) {
        PGX_ERROR("✗ NO FUNCTIONS found after Standard→LLVM lowering!");
    }
    
    // THE CRITICAL ASSERTION
    if (mainFuncAfter) {
        PGX_INFO("✓ SUCCESS: 'main' function PRESERVED through Standard→LLVM lowering");
        PGX_INFO("✓ This means PostgreSQL's JIT error is NOT in the lowering pipeline");
        PGX_INFO("✓ The issue must be in JIT execution engine or module passing");
    } else {
        PGX_ERROR("✗ FAILURE: 'main' function LOST during Standard→LLVM lowering!");
        PGX_ERROR("✗ This explains PostgreSQL's 'Module does not contain main function' error");
        PGX_ERROR("✗ The Standard→LLVM lowering pass is removing the main function");
        
        FAIL() << "Main function disappeared during Standard→LLVM lowering - THIS IS THE POSTGRESQL BUG";
    }
    
    EXPECT_NE(mainFuncAfter, nullptr) << "Main function must survive Standard→LLVM lowering";
}

// Test different function names to see if there's a naming pattern issue
TEST_F(FunctionPreservationTest, TestDifferentFunctionNames) {
    PGX_INFO("=== Testing function name preservation patterns ===");
    
    std::vector<std::string> testNames = {"main", "query_main", "compiled_query", "foo"};
    
    for (const auto& funcName : testNames) {
        PGX_INFO("Testing function name: '" + funcName + "'");
        
        // Create fresh module for each test
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
        
        // Verify function exists before lowering
        auto funcBefore = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        ASSERT_NE(funcBefore, nullptr) << "Function " << funcName << " should exist before lowering";
        
        // Apply Standard→LLVM lowering
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        auto result = pm.run(module);
        ASSERT_TRUE(mlir::succeeded(result)) << "Lowering should succeed for " << funcName;
        
        // Check if function survives
        auto funcAfter = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        if (funcAfter) {
            PGX_INFO("✓ Function '" + funcName + "' PRESERVED");
        } else {
            PGX_ERROR("✗ Function '" + funcName + "' LOST");
            
            // Debug: what functions exist instead?
            module->walk([&](mlir::func::FuncOp f) {
                PGX_INFO("  Found instead: " + std::string(f.getName()));
            });
        }
        
        EXPECT_NE(funcAfter, nullptr) << "Function " << funcName << " should survive lowering";
    }
}

// Test the exact sequence PostgreSQL uses but stop before JIT
TEST_F(FunctionPreservationTest, PostgreSQLExactSequenceWithoutJIT) {
    PGX_INFO("=== Testing PostgreSQL's exact lowering sequence (without JIT) ===");
    
    // Initialize exactly like PostgreSQL does
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    auto* utilDialect = context->getOrLoadDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr);
    utilDialect->getFunctionHelper().setParentModule(module);
    
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create "main" function exactly like postgresql_ast_translator.cpp does
    auto queryFuncType = builder.getFunctionType({}, {builder.getI32Type()});
    auto queryFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", queryFuncType);
    
    auto* entryBlock = queryFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add simple operations
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    PGX_INFO("✓ Created module with 'main' function (like PostgreSQL AST translator)");
    
    // Phase 3c: Standard→LLVM lowering (where PostgreSQL gets the warning)
    {
        PGX_INFO("Executing Phase 3c: Standard→LLVM lowering");
        
        // Verify function exists before this phase
        auto mainBefore = module.lookupSymbol<mlir::func::FuncOp>("main");
        ASSERT_NE(mainBefore, nullptr) << "main should exist before Phase 3c";
        
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        
        auto result = pm.run(module);
        ASSERT_TRUE(mlir::succeeded(result)) << "Phase 3c should succeed";
        PGX_INFO("✓ Phase 3c completed: Standard MLIR successfully lowered to LLVM IR");
        
        // CRITICAL CHECK: Does main exist after Phase 3c?
        auto mainAfter = module.lookupSymbol<mlir::func::FuncOp>("main");
        if (mainAfter) {
            PGX_INFO("✓ 'main' function EXISTS after Phase 3c - not a lowering issue");
            PGX_INFO("✓ PostgreSQL's error must be in Phase 4 (JIT execution setup)");
        } else {
            PGX_ERROR("✗ 'main' function MISSING after Phase 3c - THIS IS THE BUG");
            PGX_ERROR("✗ Standard→LLVM lowering is removing the main function");
        }
        
        EXPECT_NE(mainAfter, nullptr) << "main function must survive Phase 3c";
    }
    
    // If we get here with main function intact, the PostgreSQL issue is in JIT setup
    PGX_INFO("If main function survived Phase 3c, PostgreSQL issue is in JIT engine setup");
}

} // namespace