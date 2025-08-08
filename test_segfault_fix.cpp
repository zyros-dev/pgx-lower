#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "execution/logging.h"

using namespace mlir;

TEST(SegfaultFixTest, SimpleRelAlgToDBPass) {
    PGX_DEBUG("Testing simplified RelAlgToDB pass for segfault fix");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_segfault", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create simple RelAlg operations
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(12345)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("id"));
    auto columnsArrayAttr = builder.getArrayAttr(columnAttrs);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnsArrayAttr
    );
    
    // Create return
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    PGX_DEBUG("Test completed without segfault!");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}