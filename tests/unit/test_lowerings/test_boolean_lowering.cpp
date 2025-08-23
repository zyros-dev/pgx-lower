#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"

using namespace mlir;

class BooleanLoweringTest : public ::testing::Test {
protected:
    MLIRContext context;
    OpBuilder builder;
    ModuleOp module;

    BooleanLoweringTest() : builder(&context) {
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        
        module = ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToStart(module.getBody());
    }

    bool runDBToStdPass() {
        PassManager pm(&context);
        pm.addPass(mlir::db::createLowerToStdPass());
        return succeeded(pm.run(module));
    }

    std::string moduleToString() {
        std::string str;
        llvm::raw_string_ostream os(str);
        module.print(os);
        return str;
    }
};

TEST_F(BooleanLoweringTest, NotOpLowering) {
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_not",
        builder.getFunctionType({builder.getI1Type()}, {builder.getI1Type()}));
    
    auto* block = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    auto arg = block->getArgument(0);
    auto notOp = builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), arg);
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), notOp.getResult());
    
    ASSERT_TRUE(runDBToStdPass()) << "DBToStd pass failed";
    
    std::string mlir = moduleToString();
    EXPECT_TRUE(mlir.find("arith.cmpi eq") != std::string::npos) 
        << "Expected arith.cmpi eq (comparison with false) in lowered MLIR";
    EXPECT_FALSE(mlir.find("db.not") != std::string::npos) 
        << "db.not should be lowered";
}

TEST_F(BooleanLoweringTest, AndOpLowering) {
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_and",
        builder.getFunctionType({builder.getI1Type(), builder.getI1Type()}, {builder.getI1Type()}));
    
    auto* block = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    auto arg0 = block->getArgument(0);
    auto arg1 = block->getArgument(1);
    auto andOp = builder.create<mlir::db::AndOp>(
        builder.getUnknownLoc(), builder.getI1Type(), ValueRange{arg0, arg1});
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), andOp.getResult());
    
    ASSERT_TRUE(runDBToStdPass()) << "DBToStd pass failed";
    
    std::string mlir = moduleToString();
    EXPECT_TRUE(mlir.find("arith.andi") != std::string::npos) 
        << "Expected arith.andi in lowered MLIR";
    EXPECT_FALSE(mlir.find("db.and") != std::string::npos) 
        << "db.and should be lowered";
    EXPECT_FALSE(mlir.find("arith.select") != std::string::npos)
        << "Should not have arith.select operations in simple boolean lowering";
}

TEST_F(BooleanLoweringTest, OrOpLowering) {
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_or",
        builder.getFunctionType({builder.getI1Type(), builder.getI1Type()}, {builder.getI1Type()}));
    
    auto* block = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    auto arg0 = block->getArgument(0);
    auto arg1 = block->getArgument(1);
    auto orOp = builder.create<mlir::db::OrOp>(
        builder.getUnknownLoc(), builder.getI1Type(), ValueRange{arg0, arg1});
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), orOp.getResult());
    
    ASSERT_TRUE(runDBToStdPass()) << "DBToStd pass failed";
    
    std::string mlir = moduleToString();
    EXPECT_TRUE(mlir.find("arith.ori") != std::string::npos) 
        << "Expected arith.ori in lowered MLIR";
    EXPECT_FALSE(mlir.find("db.or") != std::string::npos) 
        << "db.or should be lowered";
    EXPECT_FALSE(mlir.find("arith.select") != std::string::npos)
        << "Should not have arith.select operations in simple boolean lowering";
}

TEST_F(BooleanLoweringTest, ComplexBooleanExpression) {
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_complex",
        builder.getFunctionType({builder.getI1Type(), builder.getI1Type(), builder.getI1Type()}, 
                                {builder.getI1Type()}));
    
    auto* block = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
    
    auto a = block->getArgument(0);
    auto b = block->getArgument(1);
    auto c = block->getArgument(2);
    
    auto andOp = builder.create<mlir::db::AndOp>(
        builder.getUnknownLoc(), builder.getI1Type(), ValueRange{a, b});
    auto notOp = builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), c);
    auto orOp = builder.create<mlir::db::OrOp>(
        builder.getUnknownLoc(), builder.getI1Type(), 
        ValueRange{andOp.getResult(), notOp.getResult()});
    
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), orOp.getResult());
    
    ASSERT_TRUE(runDBToStdPass()) << "DBToStd pass failed for complex expression";
    
    std::string mlir = moduleToString();
    EXPECT_TRUE(mlir.find("arith.andi") != std::string::npos) 
        << "Expected arith.andi for AND operation";
    EXPECT_TRUE(mlir.find("arith.cmpi eq") != std::string::npos) 
        << "Expected arith.cmpi eq for NOT operation (comparison with false)";
    EXPECT_TRUE(mlir.find("arith.ori") != std::string::npos) 
        << "Expected arith.ori for OR operation";
    
    EXPECT_FALSE(mlir.find("db.and") != std::string::npos)
        << "db.and should be lowered";
    EXPECT_FALSE(mlir.find("db.or") != std::string::npos) 
        << "db.or should be lowered";
    EXPECT_FALSE(mlir.find("db.not") != std::string::npos) 
        << "db.not should be lowered";
    
    EXPECT_FALSE(mlir.find("arith.select") != std::string::npos)
        << "Should not have arith.select operations";
    EXPECT_FALSE(mlir.find("util.unpack") != std::string::npos) 
        << "Should not have util.unpack operations";
    EXPECT_FALSE(mlir.find("util.pack") != std::string::npos) 
        << "Should not have util.pack operations";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}