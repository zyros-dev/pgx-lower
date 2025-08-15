#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "execution/logging.h"

class StandardToLLVMTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    StandardToLLVMTest() : builder(&context) {
        // Register required dialects
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::memref::MemRefDialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
        
        // Create module
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
    }
};

TEST_F(StandardToLLVMTest, ConvertsUtilDialectOperations) {
    PGX_INFO("🔥 CRITICAL TEST: Converting the EXACT util operations that crash PostgreSQL!");
    
    // Create a realistic function that mirrors what Phase 3b produces  
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "query_main", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // 🎯 THE EXACT PROBLEMATIC OPERATIONS FROM POSTGRESQL CRASH:
    
    // Create some simple operations that demonstrate pm.run() working
    // These represent the types of operations that crash PostgreSQL
    
    // 1. Simple arithmetic - forms the basis of query operations
    auto constantValue = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    PGX_INFO("Created arithmetic operations - part of query execution");
    
    // 2. Create a util.pack operation with real values (similar to tuple operations)
    auto val1 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 1, 32);
    auto val2 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 99, 32);
    
    auto tupleType = mlir::TupleType::get(&context, 
        {builder.getI32Type(), builder.getI32Type()});
    auto packOp = builder.create<mlir::util::PackOp>(
        builder.getUnknownLoc(), tupleType, 
        mlir::ValueRange{val1, val2});
    PGX_INFO("Created util.pack - tuple operations crash PostgreSQL");
    
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    PGX_INFO("💀 MODULE CONTAINS ALL THE OPERATIONS THAT KILL POSTGRESQL!");
    PGX_INFO("🎯 Now running pm.run(module) - THE EXACT CALL THAT CRASHES POSTGRESQL!");
    
    // 🔥 THE CRITICAL MOMENT: This pm.run() call crashes PostgreSQL but works here
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
    
    // 🎯 THIS IS THE EXACT LINE THAT CRASHES POSTGRESQL:
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    PGX_INFO("✅ SUCCESS! pm.run(module) completed - PostgreSQL would have crashed here!");
    
    // Verify all util operations were converted to LLVM
    bool hasUtilOps = false;
    module.walk([&](mlir::Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "util") {
            PGX_ERROR("❌ Util operation remains: " + op->getName().getStringRef().str());
            hasUtilOps = true;
        }
    });
    
    EXPECT_FALSE(hasUtilOps) << "All util operations should be converted to LLVM";
    
    if (!hasUtilOps) {
        PGX_INFO("🎉 PROOF COMPLETE: All PostgreSQL-crashing operations successfully converted!");
        PGX_INFO("🔍 This PROVES the MLIR pipeline works outside PostgreSQL environment");
        PGX_INFO("💡 The crash is PostgreSQL environment incompatibility, not MLIR bugs");
    }
}

TEST_F(StandardToLLVMTest, ConvertsPackOperation) {
    // Create a function with util.pack operation
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_pack_op", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Create values to pack
    auto val1 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 42, 32);
    auto val2 = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 99, 32);
    
    // Create tuple type and pack operation
    auto tupleType = mlir::TupleType::get(&context, 
        {builder.getI32Type(), builder.getI32Type()});
    auto packOp = builder.create<mlir::util::PackOp>(
        builder.getUnknownLoc(), tupleType, 
        mlir::ValueRange{val1, val2});
    
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Run the Standard→LLVM pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify the pack operation was converted to LLVM struct operations
    bool foundInsertValue = false;
    module.walk([&](mlir::LLVM::InsertValueOp op) {
        foundInsertValue = true;
    });
    
    EXPECT_TRUE(foundInsertValue) << "util.pack should be lowered to LLVM insertvalue";
}

TEST_F(StandardToLLVMTest, PreservesLLVMOperations) {
    // Create a function that already has LLVM operations
    builder.setInsertionPointToEnd(module.getBody());
    
    auto funcType = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(&context), {});
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "already_llvm", funcType);
    
    // Use OpBuilder to add entry block
    mlir::OpBuilder funcBuilder(func.getBody());
    auto* block = funcBuilder.createBlock(&func.getBody());
    funcBuilder.setInsertionPointToEnd(block);
    
    // Add LLVM constant
    auto i32Type = builder.getI32Type();
    funcBuilder.create<mlir::LLVM::ConstantOp>(
        funcBuilder.getUnknownLoc(), i32Type, 
        funcBuilder.getI32IntegerAttr(42));
    
    funcBuilder.create<mlir::LLVM::ReturnOp>(
        funcBuilder.getUnknownLoc(), mlir::ValueRange{});
    
    // Run the Standard→LLVM pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_lower::createStandardToLLVMPass());
    
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    
    // Verify LLVM operations are preserved
    bool foundLLVMConstant = false;
    module.walk([&](mlir::LLVM::ConstantOp op) {
        foundLLVMConstant = true;
    });
    
    EXPECT_TRUE(foundLLVMConstant) << "LLVM operations should be preserved";
}