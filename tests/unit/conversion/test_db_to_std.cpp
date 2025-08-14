#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "execution/logging.h"

namespace {

class DBToStdTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        
        // Load all required dialects
        context->getOrLoadDialect<mlir::func::FuncDialect>();
        context->getOrLoadDialect<mlir::arith::ArithDialect>();
        context->getOrLoadDialect<mlir::db::DBDialect>();
        context->getOrLoadDialect<mlir::dsa::DSADialect>();
        context->getOrLoadDialect<mlir::util::UtilDialect>();
        
        // Register conversion passes
        mlir::db::registerDBConversionPasses();
        
        builder = std::make_unique<mlir::OpBuilder>(context.get());
    }

    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<mlir::OpBuilder> builder;
};

TEST_F(DBToStdTest, EmptyModulePassesSuccessfully) {
    // Create an empty module
    auto module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    // Create a simple function to ensure module is not completely empty
    builder->setInsertionPointToStart(module.getBody());
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<mlir::func::FuncOp>(
        builder->getUnknownLoc(), "test_func", funcType);
    
    // Add a return operation
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc());
    
    // Verify module is valid
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    
    // Create and run the DB→Std pass
    mlir::PassManager pm(context.get());
    pm.addPass(mlir::db::createLowerToStdPass());
    
    // This should not crash
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result));
}

TEST_F(DBToStdTest, SimpleDBOperationLowers) {
    // Create a module with a simple DB operation
    auto module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    builder->setInsertionPointToStart(module.getBody());
    
    // Create a function that uses DB operations
    auto i32Type = builder->getI32Type();
    auto funcType = builder->getFunctionType({i32Type, i32Type}, {builder->getI1Type()});
    auto func = builder->create<mlir::func::FuncOp>(
        builder->getUnknownLoc(), "compare_func", funcType);
    
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a simple DB comparison operation
    auto cmpOp = builder->create<mlir::db::CmpOp>(
        builder->getUnknownLoc(),
        builder->getI1Type(),
        mlir::db::DBCmpPredicateAttr::get(context.get(), mlir::db::DBCmpPredicate::eq),
        entryBlock->getArgument(0),
        entryBlock->getArgument(1));
    
    builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc(), cmpOp.getResult());
    
    // Verify module is valid
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    
    // Create and run the DB→Std pass
    mlir::PassManager pm(context.get());
    pm.addPass(mlir::db::createLowerToStdPass());
    
    // This should successfully lower the DB operation
    auto result = pm.run(module);
    EXPECT_TRUE(mlir::succeeded(result));
    
    // Verify the DB operation was replaced
    bool hasDBOps = false;
    module.walk([&](mlir::db::CmpOp) { hasDBOps = true; });
    EXPECT_FALSE(hasDBOps) << "DB operations should have been lowered";
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}