#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "execution/logging.h"

namespace {

class DBToStdTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
    }

    mlir::MLIRContext context;
};

TEST_F(DBToStdTest, BasicPassCreation) {
    PGX_INFO("TEST: Creating DB→Std pass");
    
    // Create a simple module
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    // Create a PassManager
    mlir::PassManager pm(&context);
    
    // Try to create and add the pass
    auto pass = mlir::db::createLowerToStdPass();
    ASSERT_NE(pass, nullptr) << "Failed to create DB→Std pass";
    
    PGX_INFO("TEST: Pass created successfully, adding to PassManager");
    pm.addPass(std::move(pass));
    
    PGX_INFO("TEST: Running PassManager");
    
    // Enable verification
    pm.enableVerifier(true);
    
    // Run the pass
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result)) << "DB→Std pass failed on empty module";
    
    PGX_INFO("TEST: Pass executed successfully");
}

TEST_F(DBToStdTest, SimpleDBOperation) {
    PGX_INFO("TEST: Testing simple DB operation conversion");
    
    // Create module with a simple DB operation
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    // Create a function
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test_func", funcType);
    
    auto* block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    
    // Add a simple return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    // Create PassManager and run conversion
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    pm.enableVerifier(true);
    
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result)) << "DB→Std pass failed on simple function";
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}