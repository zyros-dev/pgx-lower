#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/DB/DBDialect.h"
#include "mlir/Dialect/Util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "execution/logging.h"

namespace {

class DBToStdParentModuleTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    
    void SetUp() override {
        // Load required dialects
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        
        PGX_INFO("Test setup complete with dialects loaded");
    }
};

TEST_F(DBToStdParentModuleTest, TestSetParentModuleBeforePass) {
    // Create a simple module with a function
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    // Add a simple function to the module
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    // Get UtilDialect and set parent module BEFORE running the pass
    auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr) << "UtilDialect should be loaded";
    
    // Set parent module as the DBToStd pass requires
    utilDialect->getFunctionHelper().setParentModule(module);
    PGX_INFO("Parent module set successfully before pass execution");
    
    // Create and run the DBToStd pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    // The pass should run without crashing
    auto result = pm.run(module);
    EXPECT_TRUE(succeeded(result)) << "DBToStd pass should succeed with parent module set";
    
    // Verify the module is still valid
    EXPECT_TRUE(module.verify().succeeded()) << "Module should remain valid after pass";
}

TEST_F(DBToStdParentModuleTest, TestFunctionHelperStateConsistency) {
    // Create two different modules
    mlir::OpBuilder builder(&context);
    auto module1 = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto module2 = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    auto* utilDialect = context.getLoadedDialect<mlir::util::UtilDialect>();
    ASSERT_NE(utilDialect, nullptr);
    
    // Set parent module to first module
    utilDialect->getFunctionHelper().setParentModule(module1);
    
    // Run pass on first module
    mlir::PassManager pm1(&context);
    pm1.addPass(mlir::db::createLowerToStdPass());
    EXPECT_TRUE(succeeded(pm1.run(module1)));
    
    // Now set parent module to second module
    utilDialect->getFunctionHelper().setParentModule(module2);
    
    // Run pass on second module
    mlir::PassManager pm2(&context);
    pm2.addPass(mlir::db::createLowerToStdPass());
    EXPECT_TRUE(succeeded(pm2.run(module2)));
    
    PGX_INFO("Both passes completed successfully with module switching");
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}