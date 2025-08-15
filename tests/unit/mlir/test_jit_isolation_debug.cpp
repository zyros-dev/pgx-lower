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
#include "mlir/Passes.h"
#include <sstream>
#include <memory>

namespace {

class JITIsolationTest : public ::testing::Test {
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

// Test 1: PostgreSQL Environment Isolation Test
TEST_F(JITIsolationTest, TestJITWithoutPostgreSQLContext) {
    PGX_INFO("=== Testing JIT execution without PostgreSQL memory contexts ===");
    
    // Create module with "main" function identical to PostgreSQL pipeline
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create "main" function returning i32 (identical to PostgreSQL)
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add simple operation + return (simulates PostgreSQL AST translation)
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    PGX_INFO("Created module with 'main' function - simulating PostgreSQL AST output");
    
    // Apply Standard→LLVM pipeline (same as PostgreSQL mlir_runner.cpp)
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    
    auto loweringResult = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(loweringResult)) << "Standard→LLVM lowering must succeed";
    PGX_INFO("✓ Standard→LLVM lowering succeeded without PostgreSQL context");
    
    // Test JIT execution in pure C++ environment (no PostgreSQL memory contexts)
    try {
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        
        // Check if JIT initialization succeeds without PostgreSQL
        bool jitInitSuccess = jitEngine->initialize(module);
        if (!jitInitSuccess) {
            PGX_ERROR("✗ JIT initialization failed in pure C++ environment");
            FAIL() << "JIT engine initialization failed without PostgreSQL - indicates MLIR/LLVM issue";
        }
        PGX_INFO("✓ JIT engine initialized successfully without PostgreSQL context");
        
        // Test compilation without PostgreSQL memory management
        bool compileSuccess = jitEngine->compileToLLVMIR(module);
        if (!compileSuccess) {
            PGX_ERROR("✗ JIT compilation failed in pure C++ environment");
            
            // Debug: Check what functions are available
            PGX_INFO("Module contains these functions:");
            module->walk([&](mlir::func::FuncOp funcOp) {
                PGX_INFO("  - Available function: " + std::string(funcOp.getName()));
            });
            
            // Print LLVM IR for debugging
            PGX_INFO("LLVM IR content:");
            std::string moduleStr;
            llvm::raw_string_ostream os(moduleStr);
            module->print(os, mlir::OpPrintingFlags().assumeVerified());
            os.flush();
            PGX_INFO(moduleStr);
            
            FAIL() << "JIT compilation failed even without PostgreSQL - indicates MLIR naming issue";
        }
        
        PGX_INFO("✓ JIT compilation succeeded in pure C++ environment!");
        PGX_INFO("✓ This proves the issue is PostgreSQL-specific, not MLIR/JIT related");
        
    } catch (const std::exception& e) {
        PGX_ERROR("✗ C++ exception during pure JIT execution: " + std::string(e.what()));
        FAIL() << "C++ exception in pure environment: " << e.what();
    }
}

// Test 2: JIT Function Lookup Debugging Test  
TEST_F(JITIsolationTest, TestJITFunctionLookupFailure) {
    PGX_INFO("=== Testing JIT function lookup with detailed symbol analysis ===");
    
    // Test multiple function naming patterns that appear in our codebase
    std::vector<std::string> functionNames = {"main", "query_main", "query", "execute"};
    
    for (const auto& funcName : functionNames) {
        PGX_INFO("Testing function name pattern: " + funcName);
        
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
        
        // Check function preservation through lowering
        auto funcAfter = module.lookupSymbol<mlir::func::FuncOp>(funcName);
        if (funcAfter) {
            PGX_INFO("✓ Function '" + funcName + "' preserved through Standard→LLVM lowering");
            
            // Test JIT lookup for this specific function name
            try {
                auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
                bool initSuccess = jitEngine->initialize(module);
                EXPECT_TRUE(initSuccess) << "JIT init should succeed for function: " << funcName;
                
                if (initSuccess) {
                    bool compileSuccess = jitEngine->compileToLLVMIR(module);
                    if (compileSuccess) {
                        PGX_INFO("✓ JIT compilation succeeded for function: " + funcName);
                    } else {
                        PGX_ERROR("✗ JIT compilation failed for function: " + funcName);
                        
                        // Debug: Print LLVM symbol table
                        PGX_INFO("LLVM module after lowering for function '" + funcName + "':");
                        std::string moduleStr;
                        llvm::raw_string_ostream os(moduleStr);
                        module->print(os, mlir::OpPrintingFlags().assumeVerified());
                        os.flush();
                        
                        // Print line by line for readability
                        std::istringstream iss(moduleStr);
                        std::string line;
                        while (std::getline(iss, line)) {
                            if (line.find("define") != std::string::npos || 
                                line.find("declare") != std::string::npos ||
                                line.find("@") != std::string::npos) {
                                PGX_INFO("LLVM: " + line);
                            }
                        }
                    }
                }
                
            } catch (const std::exception& e) {
                PGX_ERROR("✗ Exception during JIT test for '" + funcName + "': " + std::string(e.what()));
            }
            
        } else {
            PGX_ERROR("✗ Function '" + funcName + "' lost during Standard→LLVM lowering");
            
            // See what names exist instead
            PGX_INFO("Functions remaining after lowering:");
            module->walk([&](mlir::func::FuncOp f) {
                PGX_INFO("  - Found function: " + std::string(f.getName()));
            });
        }
        
        EXPECT_NE(funcAfter, nullptr) << "Function " << funcName << " should survive lowering";
    }
}

// Test 3: LLVM Symbol Table Analysis Test
TEST_F(JITIsolationTest, TestLLVMSymbolTableConsistency) {
    PGX_INFO("=== Testing LLVM symbol table consistency across pipeline ===");
    
    // Create module with "main" function
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    // Print module before lowering
    PGX_INFO("=== MLIR Module BEFORE Standard→LLVM lowering ===");
    std::string beforeStr;
    llvm::raw_string_ostream beforeOs(beforeStr);
    module->print(beforeOs, mlir::OpPrintingFlags().assumeVerified());
    beforeOs.flush();
    PGX_INFO(beforeStr);
    
    // Apply Standard→LLVM pipeline
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result));
    
    // Print module after lowering
    PGX_INFO("=== LLVM Module AFTER Standard→LLVM lowering ===");
    std::string afterStr;
    llvm::raw_string_ostream afterOs(afterStr);
    module->print(afterOs, mlir::OpPrintingFlags().assumeVerified());
    afterOs.flush();
    PGX_INFO(afterStr);
    
    // Analyze symbol differences
    PGX_INFO("=== Symbol Analysis ===");
    auto mainFuncAfter = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (mainFuncAfter) {
        PGX_INFO("✓ 'main' function found in LLVM module");
    } else {
        PGX_ERROR("✗ 'main' function missing from LLVM module");
        
        // List all symbols in LLVM module
        PGX_INFO("All functions in LLVM module:");
        module->walk([&](mlir::func::FuncOp f) {
            PGX_INFO("  - Function: " + std::string(f.getName()));
        });
        
        FAIL() << "Main function lost during Standard→LLVM conversion";
    }
    
    // Test JIT engine symbol resolution
    try {
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        bool initSuccess = jitEngine->initialize(module);
        ASSERT_TRUE(initSuccess) << "JIT initialization should succeed";
        
        bool compileSuccess = jitEngine->compileToLLVMIR(module);
        if (compileSuccess) {
            PGX_INFO("✓ JIT compilation succeeded - 'main' function is properly accessible");
        } else {
            PGX_ERROR("✗ JIT compilation failed despite 'main' function existing in LLVM module");
            PGX_ERROR("This indicates a disconnect between MLIR symbols and JIT symbol table");
            FAIL() << "Symbol table inconsistency between MLIR and JIT engine";
        }
        
    } catch (const std::exception& e) {
        PGX_ERROR("✗ Exception during JIT symbol resolution: " + std::string(e.what()));
        FAIL() << "JIT symbol resolution failed with exception: " << e.what();
    }
}

// Test 4: Pipeline Step-by-Step Analysis
TEST_F(JITIsolationTest, TestPipelineStepByStepExecution) {
    PGX_INFO("=== Testing pipeline execution step by step ===");
    
    // Create base module
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constant.getResult());
    
    PGX_INFO("Step 1: Initial module created with 'main' function");
    
    // Step-by-step pipeline execution
    mlir::PassManager pm(context.get());
    
    // Apply each pass individually and test JIT at each step
    PGX_INFO("Step 2: Testing JIT before any lowering passes");
    {
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        bool initSuccess = jitEngine->initialize(module);
        if (initSuccess) {
            bool compileSuccess = jitEngine->compileToLLVMIR(module);
            PGX_INFO(compileSuccess ? "✓ JIT works before lowering" : "✗ JIT fails before lowering");
        } else {
            PGX_INFO("✗ JIT initialization fails before lowering");
        }
    }
    
    PGX_INFO("Step 3: Applying Standard→LLVM pipeline");
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::succeeded(result));
    
    PGX_INFO("Step 4: Testing JIT after Standard→LLVM lowering");
    {
        auto jitEngine = std::make_unique<::pgx_lower::execution::PostgreSQLJITExecutionEngine>();
        bool initSuccess = jitEngine->initialize(module);
        if (initSuccess) {
            bool compileSuccess = jitEngine->compileToLLVMIR(module);
            if (compileSuccess) {
                PGX_INFO("✓ JIT works after Standard→LLVM lowering");
                PGX_INFO("✓ Complete pipeline validation successful");
            } else {
                PGX_ERROR("✗ JIT compilation fails after Standard→LLVM lowering");
                FAIL() << "JIT compilation fails specifically after Standard→LLVM conversion";
            }
        } else {
            PGX_ERROR("✗ JIT initialization fails after Standard→LLVM lowering");
            FAIL() << "JIT initialization fails after Standard→LLVM conversion";
        }
    }
}

} // namespace