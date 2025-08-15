#include <iostream>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Passes.h"

// Include our custom dialects
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/mlir/Passes.h"

int main() {
    std::cout << "Testing DB+DSA→Standard pipeline in isolation..." << std::endl;
    
    // Create MLIR context
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                   mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                   mlir::relalg::RelAlgDialect, mlir::db::DBDialect,
                   mlir::dsa::DSADialect, mlir::util::UtilDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    
    // Disable threading for PostgreSQL compatibility  
    context.disableMultithreading();
    std::cout << "MLIR context created with threading disabled" << std::endl;
    
    // Create a simple module 
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create function with return
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_main", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    std::cout << "Simple module created successfully" << std::endl;
    
    // Initialize UtilDialect function helper
    auto* utilDialect = context.getOrLoadDialect<mlir::util::UtilDialect>();
    if (!utilDialect) {
        std::cout << "ERROR: Failed to load UtilDialect" << std::endl;
        return 1;
    }
    std::cout << "UtilDialect loaded successfully" << std::endl;
    
    // Test: Create the exact same pipeline that crashes in PostgreSQL
    std::cout << "Creating DB+DSA→Standard pipeline..." << std::endl;
    
    mlir::PassManager pm(&context);
    pm.enableTiming();
    pm.enableStatistics();
    pm.enableVerifier(true);
    
    std::cout << "PassManager created with debugging features" << std::endl;
    
    // Call the exact same function that crashes in PostgreSQL
    std::cout << "Calling createDBDSAToStandardPipeline..." << std::endl;
    mlir::pgx_lower::createDBDSAToStandardPipeline(pm);
    
    std::cout << "Pipeline created successfully!" << std::endl;
    
    // Try to run the pipeline
    std::cout << "Running pipeline..." << std::endl;
    if (mlir::failed(pm.run(module))) {
        std::cout << "ERROR: Pipeline execution failed" << std::endl;
        return 1;
    }
    
    std::cout << "SUCCESS: Pipeline executed successfully!" << std::endl;
    std::cout << "This proves the crash is PostgreSQL-context specific" << std::endl;
    
    return 0;
}