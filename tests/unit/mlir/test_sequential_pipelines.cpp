#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "execution/logging.h"

namespace {

class SequentialPipelinesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create context with all dialects
        context = std::make_shared<mlir::MLIRContext>();
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                       mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                       mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                       mlir::relalg::RelAlgDialect, mlir::db::DBDialect,
                       mlir::dsa::DSADialect, mlir::util::UtilDialect>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
    }
    
    std::shared_ptr<mlir::MLIRContext> context;
};

// Test that the StandardToLLVM pass can be created without DataLayoutAnalysis issues
TEST_F(SequentialPipelinesTest, TestStandardToLLVMPassCreation) {
    // Create a simple module with standard operations
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder builder(module.getBodyRegion());
    
    // Create a simple function with standard operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add a simple arithmetic operation
    auto c1 = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 32);
    auto c2 = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 2, 32);
    auto add = builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), c1, c2);
    
    // Add return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Run StandardToLLVM pipeline
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createStandardToLLVMPipeline(pm, true);
    
    // This should not crash with DataLayoutAnalysis issues
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result));
}

// Test the complete sequential pipeline approach
TEST_F(SequentialPipelinesTest, TestSequentialPipelineExecution) {
    // Create a module that simulates what would come from RelAlg→DB
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    
    // Initialize UtilDialect function helper
    auto* utilDialect = context->getOrLoadDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr);
    utilDialect->getFunctionHelper().setParentModule(module);
    
    // Create a function that represents DB operations
    mlir::OpBuilder builder(module.getBodyRegion());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "query_main", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add return (simulating empty query for now)
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Test Phase 2: DB+DSA→Standard
    {
        PGX_INFO("Testing Phase 2: DB+DSA→Standard pipeline");
        mlir::PassManager pm2(context.get());
        mlir::pgx_lower::createDBDSAToStandardPipeline(pm2, true);
        auto result = pm2.run(module);
        EXPECT_TRUE(mlir::succeeded(result));
    }
    
    // Test Phase 3: Standard→LLVM
    {
        PGX_INFO("Testing Phase 3: Standard→LLVM pipeline");
        mlir::PassManager pm3(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm3, true);
        auto result = pm3.run(module);
        EXPECT_TRUE(mlir::succeeded(result));
    }
    
    // Verify final module is valid LLVM
    EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
}

} // namespace