#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include <sstream>

namespace {

// Test the fixed RelAlgToDB pass that properly handles operation lifecycle
TEST(RelAlgToDBFixedTest, ProperOperationLifecycle) {
    mlir::MLIRContext context;
    
    // Register required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    
    // Parse MLIR input instead of building programmatically to avoid segfault
    const char* mlirInput = R"mlir(
        func.func @test1_query() -> !relalg.table {
            %0 = "relalg.basetable"() {table_name = "test", table_oid = 12345 : i64} : () -> !relalg.tuplestream
            %1 = "relalg.materialize"(%0) {columns = ["*"]} : (!relalg.tuplestream) -> !relalg.table
            return %1 : !relalg.table
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirInput, &context);
    ASSERT_TRUE(module) << "Failed to parse MLIR module";
    
    // Get the function
    auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("test1_query");
    ASSERT_TRUE(funcOp) << "Failed to find test function";
    
    // Count operations before pass
    int relalgOpsBefore = 0;
    funcOp.walk([&](mlir::Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            relalgOpsBefore++;
        }
    });
    EXPECT_EQ(relalgOpsBefore, 2) << "Should have 2 RelAlg operations before pass";
    
    // Run the pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should succeed";
    
    // Verify results
    int dbOpsAfter = 0;
    int dsaOpsAfter = 0;
    int relalgOpsAfter = 0;
    
    funcOp.walk([&](mlir::Operation* op) {
        auto dialectName = op->getDialect() ? op->getDialect()->getNamespace() : "";
        if (dialectName == "db") dbOpsAfter++;
        else if (dialectName == "dsa") dsaOpsAfter++;
        else if (dialectName == "relalg") relalgOpsAfter++;
    });
    
    EXPECT_GT(dbOpsAfter, 0) << "Should generate DB operations for table access (Phase 4c-4)";
    EXPECT_GT(dsaOpsAfter, 0) << "Should generate DSA operations for result building";
    EXPECT_EQ(relalgOpsAfter, 0) << "All RelAlg operations should be erased";
    
    // Verify function type is correctly updated by the pass
    auto funcType = funcOp.getFunctionType();
    ASSERT_EQ(funcType.getNumResults(), 1) << "Function should have one result";
    // The pass should update function types from RelAlg to DSA
    EXPECT_TRUE(funcType.getResult(0).isa<pgx::mlir::dsa::TableType>()) 
        << "Function type should be updated to DSA table type";
    
    // Module cleanup handled automatically by MLIR context
}

// Test that the pass handles multiple MaterializeOps correctly
TEST(RelAlgToDBFixedTest, MultipleMaterializeOps) {
    mlir::MLIRContext context;
    
    // Register required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    
    // Create module with builder API but use carefully
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    {
        // Use a scope to control builder lifetime
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        
        // Create function
        auto funcType = builder.getFunctionType({}, {});
        auto funcOp = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), "test_multiple", funcType);
        
        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        // Create two separate pipelines
        auto baseTable1 = builder.create<pgx::mlir::relalg::BaseTableOp>(
            builder.getUnknownLoc(), builder.getStringAttr("table1"), 11111);
        
        auto materialize1 = builder.create<pgx::mlir::relalg::MaterializeOp>(
            builder.getUnknownLoc(), 
            pgx::mlir::relalg::TableType::get(&context),
            baseTable1.getResult(), 
            builder.getArrayAttr({builder.getStringAttr("*")}));
        
        auto baseTable2 = builder.create<pgx::mlir::relalg::BaseTableOp>(
            builder.getUnknownLoc(), builder.getStringAttr("table2"), 22222);
        
        auto materialize2 = builder.create<pgx::mlir::relalg::MaterializeOp>(
            builder.getUnknownLoc(),
            pgx::mlir::relalg::TableType::get(&context),
            baseTable2.getResult(), 
            builder.getArrayAttr({builder.getStringAttr("*")}));
        
        // Just return without using the results
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        
        // Run the pass
        mlir::PassManager pm(&context);
        pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
        
        auto result = pm.run(funcOp);
        ASSERT_TRUE(succeeded(result)) << "Pass should succeed with multiple MaterializeOps";
        
        // Verify all RelAlg operations were erased
        int relalgOps = 0;
        funcOp.walk([&](mlir::Operation* op) {
            if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
                relalgOps++;
            }
        });
        
        EXPECT_EQ(relalgOps, 0) << "All RelAlg operations should be erased";
    }
    
    // Module cleanup handled automatically by MLIR context
}

} // namespace