#include <gtest/gtest.h>
#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "execution/logging.h"

using namespace mlir;

class StandardToLLVMTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        context.getOrLoadDialect<util::UtilDialect>();
        context.getOrLoadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext context;
};

TEST_F(StandardToLLVMTest, ConvertsUtilOperations) {
    // Create a module with Util operations
    OpBuilder builder(&context);
    OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());

    // Create a function with Util operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_util_ops", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create some Util operations
    auto i64Type = builder.getI64Type();
    auto tupleType = builder.getTupleType({i64Type, i64Type});
    
    // Create an undef tuple
    auto undefOp = builder.create<util::UndefOp>(builder.getUnknownLoc(), tupleType);
    
    // Create constants
    auto const1 = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 42, i64Type);
    auto const2 = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 84, i64Type);
    
    // Pack values into tuple
    auto packOp = builder.create<util::PackOp>(builder.getUnknownLoc(), tupleType, 
                                               ValueRange{const1, const2});
    
    // Extract from tuple
    auto getTupleOp = builder.create<util::GetTupleOp>(builder.getUnknownLoc(), 
                                                       packOp.getResult(), 0);
    
    // Return
    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    // Run the unified StandardToLLVM pass
    PassManager pm(&context);
    pm.addPass(pgx_lower::createStandardToLLVMPass());
    
    ASSERT_TRUE(succeeded(pm.run(module.get())));
    
    // Verify all operations are converted to LLVM dialect
    bool hasNonLLVMOps = false;
    module->walk([&](Operation* op) {
        if (!isa<ModuleOp>(op) && !isa<LLVM::LLVMFuncOp>(op) && 
            op->getDialect()->getNamespace() != "llvm") {
            PGX_ERROR("Found non-LLVM operation: " + op->getName().getStringRef().str());
            hasNonLLVMOps = true;
        }
    });
    
    EXPECT_FALSE(hasNonLLVMOps) << "Module still contains non-LLVM operations after conversion";
}

TEST_F(StandardToLLVMTest, ConvertsArithOperations) {
    // Create a module with arithmetic operations
    OpBuilder builder(&context);
    OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());

    // Create a function with arithmetic operations
    auto i64Type = builder.getI64Type();
    auto funcType = builder.getFunctionType({i64Type, i64Type}, {i64Type});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "add_numbers", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Perform addition
    auto addOp = builder.create<arith::AddIOp>(builder.getUnknownLoc(), 
                                               entryBlock->getArgument(0), 
                                               entryBlock->getArgument(1));
    
    // Return result
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), addOp.getResult());

    // Run the unified StandardToLLVM pass
    PassManager pm(&context);
    pm.addPass(pgx_lower::createStandardToLLVMPass());
    
    ASSERT_TRUE(succeeded(pm.run(module.get())));
    
    // Verify conversion succeeded
    bool foundLLVMAdd = false;
    module->walk([&](LLVM::AddOp op) {
        foundLLVMAdd = true;
    });
    
    EXPECT_TRUE(foundLLVMAdd) << "Arith add operation was not converted to LLVM add";
}

TEST_F(StandardToLLVMTest, ConvertsSCFOperations) {
    // Create a module with SCF operations
    OpBuilder builder(&context);
    OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());

    // Create a function with SCF operations
    auto i64Type = builder.getI64Type();
    auto funcType = builder.getFunctionType({i64Type}, {i64Type});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_scf", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create constants
    auto zero = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 0, i64Type);
    auto one = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, i64Type);
    auto ten = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 10, i64Type);
    
    // Create SCF for loop
    auto forOp = builder.create<scf::ForOp>(builder.getUnknownLoc(), zero, ten, one, 
                                           ValueRange{entryBlock->getArgument(0)});
    
    // Add loop body
    builder.setInsertionPointToStart(forOp.getBody());
    auto sum = builder.create<arith::AddIOp>(builder.getUnknownLoc(), 
                                             forOp.getRegionIterArg(0), 
                                             forOp.getInductionVar());
    builder.create<scf::YieldOp>(builder.getUnknownLoc(), sum.getResult());
    
    // Return final result
    builder.setInsertionPointAfter(forOp);
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), forOp.getResult(0));

    // Run the unified StandardToLLVM pass
    PassManager pm(&context);
    pm.addPass(pgx_lower::createStandardToLLVMPass());
    
    ASSERT_TRUE(succeeded(pm.run(module.get())));
    
    // Verify SCF operations are converted to LLVM control flow
    bool hasSCFOps = false;
    module->walk([&](scf::ForOp op) {
        hasSCFOps = true;
    });
    
    EXPECT_FALSE(hasSCFOps) << "Module still contains SCF operations after conversion";
}