// Minimal test to isolate RelAlgToDB segfault
#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "execution/logging.h"

using namespace mlir;

TEST(MinimalSegfaultTest, IsolateIssue) {
    PGX_INFO("Starting minimal segfault isolation test");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load only required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<func::FuncDialect>();
    
    // Create minimal module
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create minimal RelAlg operations
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test"),
        builder.getI64IntegerAttr(1)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto columnAttrs = builder.getArrayAttr({builder.getStringAttr("id")});
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        UnknownLoc::get(&context),
        tableType,
        baseTableOp.getResult(),
        columnAttrs
    );
    
    builder.create<pgx::mlir::relalg::ReturnOp>(UnknownLoc::get(&context));
    
    PGX_INFO("Created minimal IR, running pass...");
    
    // Run the problematic pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    PGX_INFO("About to run pass manager...");
    LogicalResult result = pm.run(funcOp);
    PGX_INFO("Pass manager returned: " + std::string(succeeded(result) ? "success" : "failure"));
    
    // If we get here without segfault, the issue is in the test verification
    PGX_INFO("Test completed without segfault");
    
    ASSERT_TRUE(succeeded(result));
}