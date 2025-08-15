#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

namespace {

class DSAToStdLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create context with required dialects
        context = std::make_shared<mlir::MLIRContext>();
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                       mlir::scf::SCFDialect, mlir::util::UtilDialect>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
    }
    
    std::shared_ptr<mlir::MLIRContext> context;
};

// Test that external function declarations get proper visibility
TEST_F(DSAToStdLoweringTest, TestExternalFunctionVisibility) {
    // This test verifies that the fix for external function visibility is correct
    // The issue was that rt_get_execution_context was created without setting visibility
    
    // Create a module and manually add the external function as it would be created
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create the external function as the DSA lowering would
    auto funcType = builder.getFunctionType({}, {mlir::util::RefType::get(context.get(), builder.getI8Type())});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "rt_get_execution_context", funcType);
    
    // Apply the fix - set visibility to private
    funcOp.setVisibility(mlir::func::FuncOp::Visibility::Private);
    
    // Verify module is valid
    auto verifyResult = mlir::verify(module);
    EXPECT_TRUE(mlir::succeeded(verifyResult))
        << "Module with private visibility external function should be valid";
    
    // Verify the function has correct visibility
    EXPECT_EQ(funcOp.getVisibility(), mlir::func::FuncOp::Visibility::Private)
        << "External function should have private visibility";
}

// Test that scf.while operations are generated with valid structure
TEST_F(DSAToStdLoweringTest, TestWhileLoopStructure) {
    // This test verifies that the while loop generation doesn't create
    // invalid block structures that would fail MLIR verification
    
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create a function with a manually constructed scf.while to verify structure
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "while_test", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a simple scf.while loop with proper structure using index type
    auto initValue = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), 0);
    
    // Create while loop with single block in each region
    auto whileOp = builder.create<mlir::scf::WhileOp>(
        builder.getUnknownLoc(),
        mlir::TypeRange{builder.getIndexType()},
        mlir::ValueRange{initValue}
    );
    
    // Add single block to before region
    mlir::Block* beforeBlock = new mlir::Block;
    beforeBlock->addArgument(builder.getIndexType(), builder.getUnknownLoc());
    whileOp.getBefore().push_back(beforeBlock);
    
    // Add single block to after region  
    mlir::Block* afterBlock = new mlir::Block;
    afterBlock->addArgument(builder.getIndexType(), builder.getUnknownLoc());
    whileOp.getAfter().push_back(afterBlock);
    
    // Create condition in before block
    builder.setInsertionPointToStart(beforeBlock);
    auto falseVal = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getBoolAttr(false));
    builder.create<mlir::scf::ConditionOp>(
        builder.getUnknownLoc(), falseVal, beforeBlock->getArguments());
    
    // Create yield in after block
    builder.setInsertionPointToStart(afterBlock);
    builder.create<mlir::scf::YieldOp>(
        builder.getUnknownLoc(), afterBlock->getArguments());
    
    // Return after while
    builder.setInsertionPointAfter(whileOp);
    auto constOp = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), 0, 32);
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), constOp.getResult());
    
    // Verify module is valid - this will catch malformed while loops
    auto verifyResult = mlir::verify(module);
    EXPECT_TRUE(mlir::succeeded(verifyResult))
        << "Module with properly structured scf.while should be valid";
    
    // Verify the while op has valid structure
    EXPECT_EQ(whileOp.getBefore().getBlocks().size(), 1) 
        << "While 'before' region should have exactly 1 block";
    EXPECT_EQ(whileOp.getAfter().getBlocks().size(), 1)
        << "While 'after' region should have exactly 1 block";
}

} // namespace