#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "execution/logging.h"

// Test the dialect verification logic used in mlir_runner.cpp
TEST(DialectVerificationTest, VerifyLLVMDialectCheck) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect>();
    
    // Create a simple module with LLVM and non-LLVM operations
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create an LLVM function
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(&context), {}, false);
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "test_func", llvmFuncType);
    
    // Add a block to the function
    auto* entryBlock = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);
    
    // Create an LLVM return operation (should pass verification)
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    
    // Test dialect namespace checking
    bool hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            hasNonLLVMOps = true;
        }
    });
    
    // Should not have non-LLVM ops in this case
    EXPECT_FALSE(hasNonLLVMOps);
    
    // Now create a non-LLVM operation for testing
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto arithConst = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(42));
    
    // Test again - should find non-LLVM ops now
    hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            hasNonLLVMOps = true;
        }
    });
    
    // Should have found the arith.constant operation
    EXPECT_TRUE(hasNonLLVMOps);
    
    // Verify specific dialect detection
    auto* arithOp = arithConst.getOperation();
    EXPECT_NE(arithOp->getDialect(), nullptr);
    EXPECT_EQ(arithOp->getDialect()->getNamespace(), "arith");
    
    auto* llvmOp = llvmFunc.getOperation();
    EXPECT_NE(llvmOp->getDialect(), nullptr);
    EXPECT_EQ(llvmOp->getDialect()->getNamespace(), "llvm");
}

// Test edge cases in dialect verification
TEST(DialectVerificationTest, HandleNullDialect) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    // Module operations may not have a dialect
    auto* moduleOp = module.getOperation();
    
    // Test the verification logic handles null dialect gracefully
    bool isLLVMDialect = false;
    if (moduleOp->getDialect() && moduleOp->getDialect()->getNamespace() == "llvm") {
        isLLVMDialect = true;
    }
    
    // ModuleOp is a built-in op, not LLVM dialect
    EXPECT_FALSE(isLLVMDialect);
}