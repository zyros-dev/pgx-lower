#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <dialects/pg/PgDialect.h>
#include <dialects/pg/LowerPgToSubOp.h>

class LogicalOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context_.getOrLoadDialect<mlir::arith::ArithDialect>();
        context_.getOrLoadDialect<mlir::func::FuncDialect>();
        context_.getOrLoadDialect<pgx_lower::compiler::dialect::pg::PgDialect>();
    }

    mlir::MLIRContext context_;
};

TEST_F(LogicalOpsTest, PgAndOpLowering) {
    // Test that pg.and operations can be lowered
    mlir::OpBuilder builder(&context_);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    
    // Create a simple function with pg.and
    auto funcType = builder.getFunctionType({}, {builder.getI1Type()});
    auto func = builder.create<mlir::func::FuncOp>(loc, "test_and", funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create two boolean constants
    auto trueVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(true));
    auto falseVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(false));
    
    // Create pg.and operation
    auto andOp = builder.create<pgx_lower::compiler::dialect::pg::PgAndOp>(loc, builder.getI1Type(), trueVal, falseVal);
    
    // Return the result
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{andOp.getResult()});
    
    // Verify the module before lowering
    EXPECT_TRUE(module.verify().succeeded());
    
    std::cout << "Before lowering:" << std::endl;
    module.dump();
    
    // Apply pg-to-scf lowering pass
    mlir::PassManager pm(&context_);
    pm.addPass(pgx_lower::compiler::dialect::pg::createLowerPgToSubOpPass());
    
    auto result = pm.run(module);
    
    std::cout << "After lowering:" << std::endl;
    module.dump();
    
    if (result.failed()) {
        std::cout << "❌ pg.and lowering failed" << std::endl;
        EXPECT_TRUE(false) << "pg.and lowering should succeed";
    } else {
        std::cout << "✅ pg.and lowering succeeded" << std::endl;
        EXPECT_TRUE(true);
    }
    
    // Verify the module after lowering
    EXPECT_TRUE(module.verify().succeeded());
}