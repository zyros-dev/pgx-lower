#include <iostream>
#include <memory>

// Minimal MLIR includes for testing DataLayoutAnalysis
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

int main() {
    std::cout << "Testing DataLayoutAnalysis crash isolation..." << std::endl;
    
    try {
        // Minimal MLIR setup
        mlir::MLIRContext context;
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        
        // Create minimal module
        mlir::OpBuilder builder(&context);
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        std::cout << "✓ MLIR context and module created" << std::endl;
        
        // Test the suspected crash: DataLayoutAnalysis stack allocation
        std::cout << "Creating DataLayoutAnalysis (potential crash point)..." << std::endl;
        mlir::DataLayoutAnalysis analysis(module);
        std::cout << "✓ DataLayoutAnalysis created successfully" << std::endl;
        
        // Test getAtOrAbove
        std::cout << "Testing getAtOrAbove..." << std::endl;
        auto layout = analysis.getAtOrAbove(module);
        std::cout << "✓ getAtOrAbove completed successfully" << std::endl;
        
        std::cout << "=== NO CRASH - DataLayoutAnalysis works fine in isolation! ===" << std::endl;
        std::cout << "This suggests the crash is PostgreSQL-specific." << std::endl;
        
        module.erase();
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "❌ Unknown exception" << std::endl;
        return 1;
    }
}