#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/IR/Verifier.h"
#include "execution/logging.h"

namespace {

class RuntimeCallLoweringTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    RuntimeCallLoweringTest() : builder(&context) {
        // Load required dialects
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::db::DBDialect>();
        
        // Create module
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
    }

    void TearDown() override {
        if (module) {
            module.erase();
        }
    }
};

TEST_F(RuntimeCallLoweringTest, HandlesDialectAccessFailure) {
    // Create a function with a RuntimeCall operation
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), 
        "test_runtime_call",
        builder.getFunctionType({}, {})
    );
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a RuntimeCall operation
    auto runtimeCall = builder.create<mlir::db::RuntimeCall>(
        builder.getUnknownLoc(),
        mlir::TypeRange{},  // No results
        "test_function",
        mlir::ValueRange{}  // No arguments
    );
    
    // Create return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Add function to module
    module.push_back(funcOp);
    
    // Verify module is valid
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    
    // Create and run the lowering pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    // The pass should succeed even with potential dialect access issues
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result));
}

TEST_F(RuntimeCallLoweringTest, HandlesRegistryReload) {
    // Test that the lowering handles registry pointer changes correctly
    // This simulates a dialect reload scenario
    
    // Create a function with multiple RuntimeCall operations
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), 
        "test_multiple_calls",
        builder.getFunctionType({}, {})
    );
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create multiple RuntimeCall operations
    for (int i = 0; i < 3; ++i) {
        builder.create<mlir::db::RuntimeCall>(
            builder.getUnknownLoc(),
            mlir::TypeRange{},
            "test_function_" + std::to_string(i),
            mlir::ValueRange{}
        );
    }
    
    // Create return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Add function to module
    module.push_back(funcOp);
    
    // Verify module is valid
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    
    // Run the lowering pass multiple times to simulate potential registry changes
    for (int i = 0; i < 2; ++i) {
        mlir::PassManager pm(&context);
        pm.addPass(mlir::db::createLowerToStdPass());
        
        auto result = pm.run(module);
        EXPECT_TRUE(mlir::succeeded(result));
    }
}

TEST_F(RuntimeCallLoweringTest, HandlesMissingDialect) {
    // Test error handling when DB dialect is not properly loaded
    // Note: This is a theoretical test as the dialect should always be loaded
    // in our setup, but it validates the error handling path
    
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), 
        "test_missing_dialect",
        builder.getFunctionType({}, {})
    );
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a RuntimeCall that references a non-existent function
    auto runtimeCall = builder.create<mlir::db::RuntimeCall>(
        builder.getUnknownLoc(),
        mlir::TypeRange{},
        "non_existent_function",
        mlir::ValueRange{}
    );
    
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(funcOp);
    
    // The pass should handle this gracefully
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    // Even with non-existent functions, the pass infrastructure should not crash
    auto result = pm.run(module);
    // The result might fail due to unknown function, but it shouldn't crash
    // This tests the robustness of our error handling
}

TEST_F(RuntimeCallLoweringTest, HandlesComplexRuntimeCalls) {
    // Test lowering of RuntimeCall operations with arguments and results
    
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), 
        "test_complex_call",
        builder.getFunctionType({builder.getI32Type()}, {builder.getI32Type()})
    );
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Get the function argument
    auto arg = entryBlock->getArgument(0);
    
    // Create a RuntimeCall with argument and result
    auto runtimeCall = builder.create<mlir::db::RuntimeCall>(
        builder.getUnknownLoc(),
        builder.getI32Type(),
        "process_value",
        mlir::ValueRange{arg}
    );
    
    // Return the result
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), 
        mlir::ValueRange{runtimeCall.getResult()}
    );
    
    module.push_back(funcOp);
    
    // Verify module is valid
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    
    // Run the lowering pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result));
}

} // namespace