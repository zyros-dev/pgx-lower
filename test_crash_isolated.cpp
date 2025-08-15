#include <iostream>
#include <exception>

// MLIR Core
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"

// Standard Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// Our passes  
#include "pgx_lower/mlir/Passes.h"

int main() {
    std::cout << "=== CRASH ISOLATION TEST ===" << std::endl;
    
    try {
        // Test 1: Basic MLIR Context Setup
        std::cout << "TEST 1: Setting up MLIR context..." << std::endl;
        mlir::MLIRContext context;
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        std::cout << "✓ MLIR context setup successful" << std::endl;
        
        // Test 2: Create minimal module
        std::cout << "\nTEST 2: Creating minimal module..." << std::endl;
        mlir::OpBuilder builder(&context);
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        std::cout << "✓ Module created successfully" << std::endl;
        
        // Test 3: DataLayoutAnalysis isolation
        std::cout << "\nTEST 3: Testing DataLayoutAnalysis (suspected crash)..." << std::endl;
        mlir::DataLayoutAnalysis analysis(module);
        std::cout << "✓ DataLayoutAnalysis created successfully" << std::endl;
        
        auto layout = analysis.getAtOrAbove(module);
        std::cout << "✓ getAtOrAbove completed successfully" << std::endl;
        
        // Test 4: Pass creation isolation
        std::cout << "\nTEST 4: Testing pass creation (suspected crash location)..." << std::endl;
        auto pass = mlir::pgx_lower::createConvertToLLVMPass();
        std::cout << "✓ Pass created successfully" << std::endl;
        
        // Test 5: Pipeline configuration
        std::cout << "\nTEST 5: Testing pipeline configuration..." << std::endl;
        mlir::PassManager pm(&context);
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        std::cout << "✓ Pipeline configured successfully" << std::endl;
        
        module.erase();
        std::cout << "\n=== ALL TESTS PASSED - NO CRASH IN ISOLATION ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ EXCEPTION: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n❌ UNKNOWN EXCEPTION" << std::endl;
        return 1;
    }
}