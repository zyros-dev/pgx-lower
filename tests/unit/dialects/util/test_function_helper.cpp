#include "gtest/gtest.h"
#include "mlir/Dialect/util/FunctionHelper.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::util;

class FunctionHelperTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<util::UtilDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        
        // Create a module with proper structure
        module = ModuleOp::create(UnknownLoc::get(&context));
        builder = std::make_unique<OpBuilder>(module.getBodyRegion());
        
        // Create a simple function to ensure we have a valid insertion point
        auto funcType = builder->getFunctionType({}, {});
        auto func = builder->create<func::FuncOp>(
            builder->getUnknownLoc(), "test_func", funcType);
        func.addEntryBlock();
        builder->setInsertionPointToStart(func.addEntryBlock());
    }
    
    void TearDown() override {
        module.erase();
    }
    
    MLIRContext context;
    ModuleOp module;
    std::unique_ptr<OpBuilder> builder;
};

TEST_F(FunctionHelperTest, CallWithValidInsertionPoint) {
    // Test that FunctionHelper::call works with a valid insertion point
    auto funcSpec = FunctionSpec(
        "test::func",
        "_ZN4test4funcEv",
        [](MLIRContext* ctx) { return std::vector<Type>{}; },
        [](MLIRContext* ctx) { return std::vector<Type>{IntegerType::get(ctx, 32)}; },
        false
    );
    
    // This should not crash even without setParentModule being called
    auto results = FunctionHelper::call(*builder, builder->getUnknownLoc(), 
                                       funcSpec, ValueRange{});
    
    // The function should be created in the module
    auto createdFunc = module.lookupSymbol<func::FuncOp>("_ZN4test4funcEv");
    ASSERT_TRUE(createdFunc != nullptr);
}

TEST_F(FunctionHelperTest, CallWithoutInsertionPoint) {
    // Test that FunctionHelper::call handles missing insertion point gracefully
    OpBuilder builderWithoutInsertionPoint(&context);
    
    auto funcSpec = FunctionSpec(
        "test::func2",
        "_ZN4test5func2Ev",
        [](MLIRContext* ctx) { return std::vector<Type>{}; },
        [](MLIRContext* ctx) { return std::vector<Type>{IntegerType::get(ctx, 32)}; },
        false
    );
    
    // This should return empty results instead of crashing
    auto results = FunctionHelper::call(builderWithoutInsertionPoint, 
                                       builder->getUnknownLoc(), 
                                       funcSpec, ValueRange{});
    
    ASSERT_TRUE(results.empty());
}

TEST_F(FunctionHelperTest, SetParentModuleIsNoOp) {
    // Test that setParentModule is now a no-op and doesn't cause issues
    auto* utilDialect = context.getLoadedDialect<util::UtilDialect>();
    ASSERT_TRUE(utilDialect != nullptr);
    
    // This should not crash or have any effect
    utilDialect->getFunctionHelper().setParentModule(module);
    
    // Verify we can still use FunctionHelper normally
    auto funcSpec = FunctionSpec(
        "test::func3",
        "_ZN4test5func3Ev",
        [](MLIRContext* ctx) { return std::vector<Type>{}; },
        [](MLIRContext* ctx) { return std::vector<Type>{IntegerType::get(ctx, 32)}; },
        false
    );
    
    auto results = FunctionHelper::call(*builder, builder->getUnknownLoc(), 
                                       funcSpec, ValueRange{});
    
    auto createdFunc = module.lookupSymbol<func::FuncOp>("_ZN4test5func3Ev");
    ASSERT_TRUE(createdFunc != nullptr);
}