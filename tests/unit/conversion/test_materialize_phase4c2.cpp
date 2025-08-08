#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "execution/logging.h"

using namespace mlir;

// Test to verify MaterializeOp generates the correct DSA sequence
TEST(MaterializePhase4c2Test, DISABLED_VerifyDSASequence) {
    PGX_DEBUG("Testing MaterializeOp → DSA sequence generation");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_materialize_dsa", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345)
    );
    
    // Create MaterializeOp
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
    
    // Verify the DSA operation sequence
    // Should have: create_ds → ds_append → next_row → finalize
    int createDSCount = 0;
    int dsAppendCount = 0;
    int nextRowCount = 0;
    int finalizeCount = 0;
    
    funcOp.walk([&](Operation *op) {
        if (isa<pgx::mlir::dsa::CreateDSOp>(op)) {
            createDSCount++;
            PGX_DEBUG("Found dsa.create_ds");
            
            // Verify it creates a table_builder type
            auto result = op->getResult(0);
            EXPECT_TRUE(isa<pgx::mlir::dsa::TableBuilderType>(result.getType()));
        }
        else if (isa<pgx::mlir::dsa::DSAppendOp>(op)) {
            dsAppendCount++;
            PGX_DEBUG("Found dsa.ds_append");
            
            // Verify it has the table builder as first operand
            EXPECT_GE(op->getNumOperands(), 2); // builder + at least one value
        }
        else if (isa<pgx::mlir::dsa::NextRowOp>(op)) {
            nextRowCount++;
            PGX_DEBUG("Found dsa.next_row");
        }
        else if (isa<pgx::mlir::dsa::FinalizeOp>(op)) {
            finalizeCount++;
            PGX_DEBUG("Found dsa.finalize");
            
            // Verify it produces a table type
            auto result = op->getResult(0);
            EXPECT_TRUE(isa<pgx::mlir::dsa::TableType>(result.getType()));
        }
    });
    
    // Verify the exact sequence
    EXPECT_EQ(createDSCount, 1) << "Should have exactly one dsa.create_ds";
    EXPECT_EQ(dsAppendCount, 1) << "Should have exactly one dsa.ds_append";
    EXPECT_EQ(nextRowCount, 1) << "Should have exactly one dsa.next_row";
    EXPECT_EQ(finalizeCount, 1) << "Should have exactly one dsa.finalize";
    
    // Verify MaterializeOp was removed
    int materializeCount = 0;
    funcOp.walk([&](pgx::mlir::relalg::MaterializeOp op) {
        materializeCount++;
    });
    EXPECT_EQ(materializeCount, 0) << "MaterializeOp should be converted";
    
    PGX_DEBUG("MaterializeOp → DSA sequence verification completed");
}

// Test to verify mixed DB+DSA operations
TEST(MaterializePhase4c2Test, DISABLED_MixedDBDSAOperations) {
    PGX_DEBUG("Testing mixed DB+DSA operation generation");
    
    MLIRContext context;
    OpBuilder builder(&context);
    
    // Load required dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    
    // Create module and function
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_mixed_ops", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        UnknownLoc::get(&context),
        tupleStreamType,
        builder.getStringAttr("employees"),
        builder.getI64IntegerAttr(54321)
    );
    
    // Create MaterializeOp
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    llvm::SmallVector<mlir::Attribute> columnAttrs;
    columnAttrs.push_back(builder.getStringAttr("*"));
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
    
    // Verify we have both DB and DSA operations
    bool hasDBOps = false;
    bool hasDSAOps = false;
    
    funcOp.walk([&](Operation *op) {
        if (op->getDialect()->getNamespace() == "db") {
            hasDBOps = true;
            PGX_DEBUG("Found DB operation: " + op->getName().getStringRef().str());
        }
        if (op->getDialect()->getNamespace() == "dsa") {
            hasDSAOps = true;
            PGX_DEBUG("Found DSA operation: " + op->getName().getStringRef().str());
        }
    });
    
    EXPECT_TRUE(hasDBOps) << "Should have DB operations from BaseTableOp conversion";
    EXPECT_TRUE(hasDSAOps) << "Should have DSA operations from MaterializeOp conversion";
    
    PGX_DEBUG("Mixed DB+DSA operation test completed");
}

// Phase 4c-0: Placeholder test to ensure test suite runs
TEST(MaterializePhase4c2Test, PassExists) {
    PGX_DEBUG("Running PassExists test - verifying pass can be created");
    
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
    auto funcOp = builder.create<func::FuncOp>(UnknownLoc::get(&context), "test_pass_exists", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a simple return
    builder.create<func::ReturnOp>(UnknownLoc::get(&context));
    
    // Run the pass - it should succeed even as a no-op
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should succeed as no-op";
    
    PGX_DEBUG("PassExists test completed - pass infrastructure is working");
}