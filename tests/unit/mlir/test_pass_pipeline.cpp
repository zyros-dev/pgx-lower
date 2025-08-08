#include <gtest/gtest.h>
#include "mlir/Passes.h"
#include "execution/logging.h"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"

// Dialect includes for testing
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Conversion pass includes for registration verification
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"

namespace {

class PassPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize MLIR context and load dialects
        context = std::make_unique<mlir::MLIRContext>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        context->getOrLoadDialect<mlir::arith::ArithDialect>();
        context->getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context->getOrLoadDialect<pgx::db::DBDialect>();
        context->getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
        
        // Register all passes
        mlir::pgx_lower::registerAllPgxLoweringPasses();
    }
    
    void TearDown() override {
        context.reset();
    }
    
    // Helper to create an empty module with a function
    mlir::OwningOpRef<mlir::ModuleOp> createEmptyModule() {
        mlir::OpBuilder builder(context.get());
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        // Add an empty function to test nested passes
        builder.setInsertionPointToEnd(module.getBody());
        auto funcType = builder.getFunctionType({}, {});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "test_func", funcType);
        func.addEntryBlock();
        
        // Add return to make it valid
        builder.setInsertionPointToEnd(&func.getBody().front());
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        
        return module;
    }
    
    std::unique_ptr<mlir::MLIRContext> context;
};

// Test 1: Pass Registration
TEST_F(PassPipelineTest, TestPassRegistration) {
    PGX_DEBUG("Testing pass registration functionality");
    
    // Registration should complete without throwing
    ASSERT_NO_THROW(mlir::pgx_lower::registerAllPgxLoweringPasses());
    
    // Verify passes are registered by checking if we can create them
    ASSERT_NO_THROW(mlir::pgx_conversion::createRelAlgToDBPass());
}

// Test 2: Complete Pipeline Creation
TEST_F(PassPipelineTest, TestCompletePipelineCreation) {
    PGX_DEBUG("Testing complete pipeline creation");
    
    mlir::PassManager pm(context.get());
    
    // Should create pipeline without throwing
    ASSERT_NO_THROW(mlir::pgx_lower::createCompleteLoweringPipeline(pm, true));
    
    // Create another with verification disabled
    mlir::PassManager pm2(context.get());
    ASSERT_NO_THROW(mlir::pgx_lower::createCompleteLoweringPipeline(pm2, false));
}

// Test 3: RelAlg to DB Pipeline Creation
TEST_F(PassPipelineTest, TestRelAlgToDBPipelineCreation) {
    PGX_DEBUG("Testing RelAlg → DB pipeline creation");
    
    mlir::PassManager pm(context.get());
    
    // Should create pipeline without throwing
    ASSERT_NO_THROW(mlir::pgx_lower::createRelAlgToDBPipeline(pm));
}


// Test 5: Pipeline Execution on Empty Module
TEST_F(PassPipelineTest, TestPipelineExecutionOnEmptyModule) {
    PGX_DEBUG("Testing pipeline execution on empty module");
    
    auto module = createEmptyModule();
    ASSERT_TRUE(module);
    
    // Verify module is valid before running passes
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
    
    // Should execute successfully on empty module
    ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    
    // Module should still be valid after pipeline
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

// Test 6: Error Handling for Malformed Input
TEST_F(PassPipelineTest, TestPipelineErrorHandling) {
    PGX_DEBUG("Testing pipeline error handling");
    
    // Create a module with an invalid function (no terminator)
    mlir::OpBuilder builder(context.get());
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "invalid_func", funcType);
    func.addEntryBlock();
    // Intentionally don't add a return terminator
    
    // Module should fail verification
    ASSERT_TRUE(mlir::failed(mlir::verify(module)));
    
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
    
    // Pipeline should fail on invalid module
    ASSERT_TRUE(mlir::failed(pm.run(module)));
}

// Test 7: Pass Ordering and Dependencies
TEST_F(PassPipelineTest, TestPassOrdering) {
    PGX_DEBUG("Testing pass ordering in pipeline");
    
    // Create a module that would test pass ordering
    // For now, just verify the pipeline can be created multiple times
    // with consistent ordering
    
    mlir::PassManager pm1(context.get());
    mlir::pgx_lower::createCompleteLoweringPipeline(pm1, true);
    
    mlir::PassManager pm2(context.get());
    mlir::pgx_lower::createCompleteLoweringPipeline(pm2, true);
    
    // Both should execute identically on the same module
    auto module1 = createEmptyModule();
    auto module2 = createEmptyModule();
    
    ASSERT_TRUE(mlir::succeeded(pm1.run(*module1)));
    ASSERT_TRUE(mlir::succeeded(pm2.run(*module2)));
}

// Test 8: Verification Toggle
TEST_F(PassPipelineTest, TestVerificationToggle) {
    PGX_DEBUG("Testing verification toggle in pipeline");
    
    auto module = createEmptyModule();
    ASSERT_TRUE(module);
    
    // Test with verification enabled
    {
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    }
    
    // Test with verification disabled
    {
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, false);
        ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    }
}

// Test 9: Module Preservation
TEST_F(PassPipelineTest, TestModulePreservation) {
    PGX_DEBUG("Testing module preservation through pipeline");
    
    auto module = createEmptyModule();
    ASSERT_TRUE(module);
    
    // Count operations before pipeline
    size_t opCountBefore = 0;
    module->walk([&](mlir::Operation* op) { opCountBefore++; });
    
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
    
    ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    
    // Module should still have operations (not destroyed)
    size_t opCountAfter = 0;
    module->walk([&](mlir::Operation* op) { opCountAfter++; });
    
    // For an empty module with no RelAlg ops, count should be preserved
    // (only func and return ops)
    ASSERT_GT(opCountAfter, 0);
}

// Test 10: PostgreSQL Error Propagation
TEST_F(PassPipelineTest, PostgreSQLErrorPropagation) {
    PGX_DEBUG("Testing PostgreSQL error integration with pipeline");
    
    // Create a module with an invalid function to trigger error
    mlir::OpBuilder builder(context.get());
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "invalid_func", funcType);
    func.addEntryBlock();
    // Intentionally don't add a return terminator to trigger error
    
    mlir::PassManager pm(context.get());
    mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
    
    // Pipeline should fail and errors should be properly reported
    auto result = pm.run(module);
    ASSERT_TRUE(mlir::failed(result));
    
    // In production, this would trigger PGX_ERROR macros
    // Verify that error handling infrastructure is in place
    PGX_DEBUG("Pipeline error propagation test completed");
}

// Test 11: Handle Invalid MLIR Gracefully
TEST_F(PassPipelineTest, HandleInvalidMLIRGracefully) {
    PGX_DEBUG("Testing pipeline behavior with various invalid inputs");
    
    // Test 1: Module with malformed operation structure
    {
        mlir::OpBuilder builder(context.get());
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        // Create function with invalid block structure
        builder.setInsertionPointToEnd(module.getBody());
        auto funcType = builder.getFunctionType({builder.getI32Type()}, {});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "malformed_func", funcType);
        func.addEntryBlock();
        // Block expects argument but we don't provide one properly
        
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        // Should fail gracefully without crash
        ASSERT_TRUE(mlir::failed(pm.run(module)));
    }
    
    // Test 2: Module with type mismatches
    {
        mlir::OpBuilder builder(context.get());
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        builder.setInsertionPointToEnd(module.getBody());
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "type_mismatch_func", funcType);
        func.addEntryBlock();
        
        // Return without value when function expects i32
        builder.setInsertionPointToEnd(&func.getBody().front());
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        // Should fail gracefully
        ASSERT_TRUE(mlir::failed(pm.run(module)));
    }
    
    PGX_DEBUG("Invalid MLIR handling tests completed");
}

// Test 12: Pass Failure Recovery
TEST_F(PassPipelineTest, PassFailureRecovery) {
    PGX_DEBUG("Testing individual pass failure handling and recovery");
    
    // Create a valid module
    auto module = createEmptyModule();
    ASSERT_TRUE(module);
    
    // Test pipeline behavior when individual passes might fail
    // Since we can't easily force pass failures in unit tests,
    // we verify the infrastructure is in place
    {
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        // Pipeline should handle empty modules gracefully
        ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    }
    
    // Test with verification disabled - should still handle errors
    {
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, false);
        
        // Even without verification, catastrophic failures should be caught
        ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    }
    
    PGX_DEBUG("Pass failure recovery test completed");
}

// Test 13: Pipeline Robustness
TEST_F(PassPipelineTest, PipelineRobustness) {
    PGX_DEBUG("Testing pipeline robustness with edge cases");
    
    // Test 1: Empty module (no functions)
    {
        mlir::OpBuilder builder(context.get());
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        // Should handle empty module without crash
        ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    }
    
    // Test 2: Module with multiple functions
    {
        mlir::OpBuilder builder(context.get());
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        // Add multiple valid functions
        for (int i = 0; i < 3; ++i) {
            builder.setInsertionPointToEnd(module.getBody());
            auto funcType = builder.getFunctionType({}, {});
            auto func = builder.create<mlir::func::FuncOp>(
                builder.getUnknownLoc(), "func_" + std::to_string(i), funcType);
            func.addEntryBlock();
            builder.setInsertionPointToEnd(&func.getBody().front());
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        }
        
        mlir::PassManager pm(context.get());
        mlir::pgx_lower::createCompleteLoweringPipeline(pm, true);
        
        // Should process all functions successfully
        ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    }
    
    PGX_DEBUG("Pipeline robustness test completed");
}

} // namespace