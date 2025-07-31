#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/dsa/DSADialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"

using namespace pgx_lower::compiler::dialect;

TEST(SubOpSimpleTest, BasicLowering) {
    mlir::MLIRContext context;
    
    // Load all required dialects
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<relalg::RelAlgDialect>();
    context.getOrLoadDialect<subop::SubOperatorDialect>();
    context.getOrLoadDialect<db::DBDialect>();
    context.getOrLoadDialect<dsa::DSADialect>();
    context.getOrLoadDialect<util::UtilDialect>();
    context.getOrLoadDialect<tuples::TupleStreamDialect>();
    
    // Create a minimal module with SubOp operations already
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple SubOp execution group
    auto tupleStreamType = tuples::TupleStreamType::get(&context);
    auto execGroup = builder.create<subop::ExecutionGroupOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        mlir::ValueRange{});
    
    // Add a simple body
    auto* block = new mlir::Block;
    execGroup.getSubOps().push_back(block);
    builder.setInsertionPointToEnd(block);
    
    // Create a simple constant value
    auto i32Type = builder.getI32Type();
    auto constant = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 42, i32Type);
    
    // Create return
    builder.create<subop::ExecutionGroupReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    
    std::cout << "Module before lowering:\n";
    module.dump();
    
    // Test just the final lowering pass
    auto pm = mlir::PassManager(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result)) << "SubOp lowering should succeed";
    
    std::cout << "Module after lowering:\n";
    module.dump();
}

TEST(SubOpSimpleTest, IndividualPasses) {
    mlir::MLIRContext context;
    
    // Load all required dialects
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<relalg::RelAlgDialect>();
    context.getOrLoadDialect<subop::SubOperatorDialect>();
    context.getOrLoadDialect<db::DBDialect>();
    context.getOrLoadDialect<dsa::DSADialect>();
    context.getOrLoadDialect<util::UtilDialect>();
    context.getOrLoadDialect<tuples::TupleStreamDialect>();
    
    // Test individual passes to find which works
    std::vector<std::pair<std::string, std::function<std::unique_ptr<mlir::Pass>()>>> passes = {
        {"GlobalOpt", []() { return subop::createGlobalOptPass(); }},
        {"FoldColumns", []() { return subop::createFoldColumnsPass(); }},
        {"NormalizeSubOp", []() { return subop::createNormalizeSubOpPass(); }},
        {"Canonicalizer", []() { return mlir::createCanonicalizerPass(); }},
        {"CSE", []() { return mlir::createCSEPass(); }}
    };
    
    for (const auto& [name, createPass] : passes) {
        // Create a fresh module for each test
        mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
        mlir::OpBuilder builder(&context);
        builder.setInsertionPointToEnd(module.getBody());
        
        // Create a simple SubOp execution group
        auto tupleStreamType = tuples::TupleStreamType::get(&context);
        auto execGroup = builder.create<subop::ExecutionGroupOp>(
            builder.getUnknownLoc(),
            tupleStreamType,
            mlir::ValueRange{});
        
        // Add a simple body
        auto* block = new mlir::Block;
        execGroup.getSubOps().push_back(block);
        builder.setInsertionPointToEnd(block);
        
        // Create return
        builder.create<subop::ExecutionGroupReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
        
        std::cout << "\nTesting pass: " << name << "\n";
        
        auto pm = mlir::PassManager(&context);
        pm.addPass(createPass());
        
        auto result = pm.run(module);
        bool success = mlir::succeeded(result);
        std::cout << "Result: " << (success ? "SUCCESS" : "FAILED") << "\n";
        
        if (!success) {
            module.dump();
        }
    }
}