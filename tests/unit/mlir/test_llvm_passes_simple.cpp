#include <mlir/Pass/Pass.h>
#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>

#include "execution/logging.h"

// Simple test to isolate LLVM lowering passes
class LLVMPassesSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::cf::ControlFlowDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
    }
    
    mlir::MLIRContext context;
};

TEST_F(LLVMPassesSimpleTest, BasicFuncToLLVM) {
    PGX_INFO("TEST: Starting BasicFuncToLLVM test");
    
    // Create a simple module
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple function
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    // Create a constant and return it
    auto constant = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), mlir::ValueRange{constant});
    
    // Verify module
    ASSERT_FALSE(mlir::failed(module.verify()));
    PGX_INFO("TEST: Module created and verified");
    
    // Run just FuncToLLVM conversion
    {
        PGX_INFO("TEST: Testing FuncToLLVM pass in isolation");
        mlir::PassManager pm(&context);
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        
        auto result = pm.run(module);
        if (mlir::failed(result)) {
            PGX_ERROR("TEST: FuncToLLVM pass failed");
            FAIL() << "FuncToLLVM pass failed";
        }
        PGX_INFO("TEST: FuncToLLVM pass succeeded");
    }
    
    // Module should still verify
    ASSERT_FALSE(mlir::failed(module.verify()));
    PGX_INFO("TEST: Module verified after FuncToLLVM");
}

TEST_F(LLVMPassesSimpleTest, SequentialLLVMPasses) {
    PGX_INFO("TEST: Starting SequentialLLVMPasses test");
    
    // Create module with arithmetic operations
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    auto const1 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 10, 32);
    auto const2 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 20, 32);
    auto add = builder.create<mlir::arith::AddIOp>(
        builder.getUnknownLoc(), const1, const2);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), mlir::ValueRange{add});
    
    // Verify initial module
    ASSERT_FALSE(mlir::failed(module.verify()));
    PGX_INFO("TEST: Initial module verified");
    
    // Run passes one by one
    mlir::PassManager pm(&context);
    
    // Add each pass individually
    PGX_INFO("TEST: Adding FuncToLLVM pass");
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    
    PGX_INFO("TEST: Adding ArithToLLVM pass");
    pm.addPass(mlir::createArithToLLVMConversionPass());
    
    PGX_INFO("TEST: Adding ReconcileUnrealizedCasts pass");
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    // Run the pipeline
    PGX_INFO("TEST: Running pass pipeline");
    auto result = pm.run(module);
    
    if (mlir::failed(result)) {
        PGX_ERROR("TEST: Pass pipeline failed");
        FAIL() << "Pass pipeline failed";
    }
    
    PGX_INFO("TEST: Pass pipeline succeeded");
    
    // Verify final module
    ASSERT_FALSE(mlir::failed(module.verify()));
    PGX_INFO("TEST: Final module verified");
}