#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "execution/logging.h"

TEST(DialectVerificationTest, VerifyLLVMDialectCheck) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect>();
    
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(&context), {}, false);
    auto llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "test_func", llvmFuncType);
    
    auto* entryBlock = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);
    
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    
    bool hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            hasNonLLVMOps = true;
        }
    });
    
    EXPECT_FALSE(hasNonLLVMOps);
    
    builder.setInsertionPointToStart(module.getBody());
    auto i32Type = builder.getI32Type();
    auto arithConst = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(42));
    
    hasNonLLVMOps = false;
    module->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::ModuleOp>(op) && 
            op->getDialect() && op->getDialect()->getNamespace() != "llvm") {
            hasNonLLVMOps = true;
        }
    });
    
    EXPECT_TRUE(hasNonLLVMOps);
    
    auto* arithOp = arithConst.getOperation();
    EXPECT_NE(arithOp->getDialect(), nullptr);
    EXPECT_EQ(arithOp->getDialect()->getNamespace(), "arith");
    
    auto* llvmOp = llvmFunc.getOperation();
    EXPECT_NE(llvmOp->getDialect(), nullptr);
    EXPECT_EQ(llvmOp->getDialect()->getNamespace(), "llvm");
}

TEST(DialectVerificationTest, HandleNullDialect) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    
    auto* moduleOp = module.getOperation();
    
    bool isLLVMDialect = false;
    if (moduleOp->getDialect() && moduleOp->getDialect()->getNamespace() == "llvm") {
        isLLVMDialect = true;
    }
    
    EXPECT_FALSE(isLLVMDialect);
}