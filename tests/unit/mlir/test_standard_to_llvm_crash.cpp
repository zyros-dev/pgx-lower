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

#include "mlir/Passes.h"
#include "execution/logging.h"

// Test specifically for Standard→LLVM lowering crash
class StandardToLLVMTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::scf::SCFDialect>();
        context.loadDialect<mlir::cf::ControlFlowDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
    }
    
    mlir::MLIRContext context;
};

TEST_F(StandardToLLVMTest, SimpleStandardToLLVM) {
    PGX_INFO("TEST: Starting SimpleStandardToLLVM test");
    
    // Create a simple module with standard operations
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple function with arithmetic operations
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    
    // Create a simple arithmetic operation
    auto const1 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    auto const2 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 10, 32);
    auto add = builder.create<mlir::arith::AddIOp>(
        builder.getUnknownLoc(), const1, const2);
    
    // Return the result
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), mlir::ValueRange{add});
    
    // Verify module before lowering
    if (mlir::failed(module.verify())) {
        PGX_ERROR("TEST: Module verification failed before lowering");
        FAIL() << "Module verification failed before lowering";
    }
    PGX_INFO("TEST: Module verified before lowering");
    
    // Try to run Standard→LLVM lowering
    PGX_INFO("TEST: Creating PassManager for Standard→LLVM lowering");
    mlir::PassManager pm(&context);
    
    try {
        PGX_INFO("TEST: Calling createStandardToLLVMPipeline");
        mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
        PGX_INFO("TEST: createStandardToLLVMPipeline completed successfully");
    } catch (const std::exception& e) {
        PGX_ERROR("TEST: Exception in createStandardToLLVMPipeline: " + std::string(e.what()));
        FAIL() << "Exception in createStandardToLLVMPipeline: " << e.what();
    } catch (...) {
        PGX_ERROR("TEST: Unknown exception in createStandardToLLVMPipeline");
        FAIL() << "Unknown exception in createStandardToLLVMPipeline";
    }
    
    PGX_INFO("TEST: Running PassManager");
    auto result = pm.run(module);
    
    if (mlir::failed(result)) {
        PGX_ERROR("TEST: Standard→LLVM lowering failed");
        FAIL() << "Standard→LLVM lowering failed";
    }
    
    PGX_INFO("TEST: Standard→LLVM lowering completed successfully");
    
    // Verify module after lowering
    if (mlir::failed(module.verify())) {
        PGX_ERROR("TEST: Module verification failed after lowering");
        FAIL() << "Module verification failed after lowering";
    }
    PGX_INFO("TEST: Module verified after lowering");
    
    // Check that operations were lowered to LLVM
    bool hasLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (op->getDialect() && 
            op->getDialect()->getNamespace() == "llvm") {
            hasLLVMOps = true;
        }
    });
    
    EXPECT_TRUE(hasLLVMOps) << "Module should contain LLVM operations after lowering";
    PGX_INFO("TEST: Test completed successfully");
}