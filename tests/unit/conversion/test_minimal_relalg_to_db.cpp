#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

// Minimal test to verify RelAlgToDB pass runs without segfault

namespace {

class MinimalRelAlgToDBTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    MinimalRelAlgToDBTest() : builder(&context) {
        // Register all required dialects
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::scf::SCFDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToStart(module.getBody());
    }
};

TEST_F(MinimalRelAlgToDBTest, PassRunsWithoutCrash) {
    // Create minimal IR
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {relAlgTableType});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "minimal_query", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create minimal pipeline
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), materializeOp.getResult());
    
    // Run the pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should succeed";
    
    // If we get here without segfault, the test passes
    SUCCEED() << "Pass completed without segfault";
}

TEST_F(MinimalRelAlgToDBTest, EmptyFunctionHandling) {
    // Test with empty function
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "empty_query", funcType);
    
    funcOp.addEntryBlock();
    
    // Run the pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "Pass should handle empty functions";
}

} // namespace