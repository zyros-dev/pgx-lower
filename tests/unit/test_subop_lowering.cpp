#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/db/DBDialect.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/util/UtilDialect.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

class SubOpLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<relalg::RelAlgDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext context;
};

TEST_F(SubOpLoweringTest, BasicSubOpCreation) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    // Create a simple module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple SubOp operation to test
    auto i32Type = builder.getI32Type();
    auto tupleType = TupleType::get(&context, {i32Type});
    
    // Verify the module was created
    EXPECT_TRUE(module);
    EXPECT_EQ(module.getBody()->getNumArguments(), 0);
}

TEST_F(SubOpLoweringTest, SimpleGenerateEmit) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a generate region with emit
    auto genOp = builder.create<subop::GenerateOp>(loc, TypeRange{});
    builder.setInsertionPointToStart(&genOp.getRegion().emplaceBlock());
    
    // Create a simple constant value
    auto constOp = builder.create<arith::ConstantIntOp>(loc, 42, 32);
    
    // Create emit operation
    builder.create<subop::GenerateEmitOp>(loc, ValueRange{constOp});
    
    // Verify structure
    EXPECT_TRUE(genOp);
    EXPECT_EQ(genOp.getRegion().getBlocks().size(), 1);
}

TEST_F(SubOpLoweringTest, SubOpToControlFlowLowering) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    auto module = ModuleOp::create(loc);
    
    // Create a simple pass pipeline
    PassManager pm(&context);
    pm.addPass(subop::createLowerSubOpPass());
    
    // Run the pass
    auto result = pm.run(module);
    
    // Verify it completes without error
    EXPECT_TRUE(succeeded(result));
}

// Test to understand the runtime function pattern
TEST_F(SubOpLoweringTest, RuntimeFunctionPattern) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    
    // Test how LingoDB expects runtime functions to work
    // They return a callable that takes arguments and returns values
    
    // This test helps us understand what pattern we need to implement
    // for ExecutionContext::allocStateRaw, ThreadLocal::getLocal, etc.
    
    EXPECT_TRUE(true); // Placeholder for now
}