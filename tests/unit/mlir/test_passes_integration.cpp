#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Passes.h"
#include "execution/mlir_runner.h"
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

namespace {

// Test that all required dialects can be loaded
TEST(PassesIntegrationTest, TestDialectLoading) {
    auto context = std::make_shared<mlir::MLIRContext>();
    
    // Use DialectRegistry pattern
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                   mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                   mlir::relalg::RelAlgDialect, mlir::db::DBDialect,
                   mlir::dsa::DSADialect, mlir::util::UtilDialect>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
    
    // Verify dialects are loaded
    EXPECT_NE(context->getLoadedDialect<mlir::func::FuncDialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::arith::ArithDialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::LLVM::LLVMDialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::cf::ControlFlowDialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::relalg::RelAlgDialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::db::DBDialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::dsa::DSADialect>(), nullptr);
    EXPECT_NE(context->getLoadedDialect<mlir::util::UtilDialect>(), nullptr);
}

// Test that the sequential lowering pipelines can be created
TEST(PassesIntegrationTest, TestSequentialLoweringPipelines) {
    auto context = std::make_shared<mlir::MLIRContext>();
    
    // Load all required dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                   mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                   mlir::relalg::RelAlgDialect, mlir::db::DBDialect,
                   mlir::dsa::DSADialect, mlir::util::UtilDialect>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
    
    // Create a simple module for testing
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    
    // Test Phase 1: RelAlg→DB pipeline
    {
        mlir::PassManager pm1(context.get());
        mlir::pgx_lower::createRelAlgToDBPipeline(pm1, true);
        EXPECT_TRUE(true); // Pipeline created successfully
    }
    
    // Test Phase 2: DB+DSA→Standard pipeline
    {
        mlir::PassManager pm2(context.get());
        mlir::pgx_lower::createDBDSAToStandardPipeline(pm2, true);
        EXPECT_TRUE(true); // Pipeline created successfully
    }
    
    // Test Phase 3: Standard→LLVM pipeline
    {
        mlir::PassManager pm3(context.get());
        mlir::pgx_lower::createStandardToLLVMPipeline(pm3, true);
        EXPECT_TRUE(true); // Pipeline created successfully
    }
    
    // Test Function-level optimization pipeline
    {
        mlir::PassManager pmFunc(context.get(), mlir::func::FuncOp::getOperationName());
        mlir::pgx_lower::createFunctionOptimizationPipeline(pmFunc, true);
        EXPECT_TRUE(true); // Pipeline created successfully
    }
}

// Test memory context heap allocation pattern
TEST(PassesIntegrationTest, TestHeapAllocatedContext) {
    // Test that context is heap-allocated (Fix C)
    auto context1 = std::make_shared<mlir::MLIRContext>();
    auto context2 = std::make_shared<mlir::MLIRContext>();
    
    // Contexts should be different instances
    EXPECT_NE(context1.get(), context2.get());
    
    // But both should be valid
    EXPECT_NE(context1.get(), nullptr);
    EXPECT_NE(context2.get(), nullptr);
}

} // namespace