// RESTORED: DSA operations have been restored in Phase 4d-1
// This test verifies the RelAlgToDB pass with DSA table building operations

#include <gtest/gtest.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBOps.h"
#include "pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "execution/logging.h"

using namespace mlir;

class RelAlgToDBPhase4c4Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Load required dialects
        context.loadDialect<mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<scf::SCFDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
};

// Test the RelAlgToDB pass generates DSA operations
TEST_F(RelAlgToDBPhase4c4Test, DISABLED_GeneratesDSAOperations) {
    PGX_DEBUG("Testing RelAlgToDB pass generates DSA table building operations");
    
    // Create module and function
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToStart(module.getBody());
    
    // Function starts with RelAlg table type
    auto tableType = mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_dsa_generation", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp - simplified version without column metadata
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<mlir::relalg::BaseTableOp>(
        loc,
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12345)  // table_oid
    );
    
    // Create MaterializeOp with column names
    auto columnsAttr = builder->getArrayAttr({builder->getStringAttr("id")});
    auto materializeOp = builder->create<mlir::relalg::MaterializeOp>(
        loc,
        tableType,
        baseTableOp.getResult(),
        columnsAttr
    );
    
    // Create return with the MaterializeOp result
    builder->create<func::ReturnOp>(loc, materializeOp.getResult());
    
    // Verify the function before pass
    ASSERT_TRUE(funcOp.verify().succeeded()) << "Function verification failed before pass";
    
    // Run the RelAlgToDB pass
    PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(module);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify DSA operations were generated
    bool foundCreateDS = false;
    bool foundDSAppend = false;
    bool foundNextRow = false;
    bool foundFinalize = false;
    
    funcOp.walk([&](Operation *op) {
        if (isa<mlir::dsa::CreateDS>(op)) {
            foundCreateDS = true;
            auto createOp = cast<mlir::dsa::CreateDS>(op);
            EXPECT_TRUE(createOp.getDs().getType().isa<mlir::dsa::TableBuilderType>())
                << "CreateDS should create a TableBuilder";
        } else if (isa<mlir::dsa::Append>(op)) {
            foundDSAppend = true;
        } else if (isa<mlir::dsa::NextRow>(op)) {
            foundNextRow = true;
        } else if (isa<mlir::dsa::Finalize>(op)) {
            foundFinalize = true;
            auto finalizeOp = cast<mlir::dsa::Finalize>(op);
            EXPECT_TRUE(finalizeOp.getRes().getType().isa<mlir::dsa::TableType>())
                << "Finalize should produce a Table";
        }
    });
    
    EXPECT_TRUE(foundCreateDS) << "Should generate dsa.create_ds";
    EXPECT_TRUE(foundDSAppend) << "Should generate dsa.ds_append";
    EXPECT_TRUE(foundNextRow) << "Should generate dsa.next_row";
    EXPECT_TRUE(foundFinalize) << "Should generate dsa.finalize";
    
    // Verify RelAlg operations were removed
    int relalgOpsCount = 0;
    funcOp.walk([&](Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            relalgOpsCount++;
        }
    });
    EXPECT_EQ(relalgOpsCount, 0) << "All RelAlg operations should be removed";
    
    // Verify function signature was updated
    auto updatedFuncType = funcOp.getFunctionType();
    EXPECT_EQ(updatedFuncType.getNumResults(), 0) << "Function should return void after conversion";
    
    PGX_DEBUG("RelAlgToDB pass correctly generates DSA operations");
}

// Test with multiple columns
TEST_F(RelAlgToDBPhase4c4Test, DISABLED_MultipleColumns) {
    PGX_DEBUG("Testing RelAlgToDB with multiple columns");
    
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToStart(module.getBody());
    
    auto tableType = mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_multiple_cols", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp with multiple columns
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<mlir::relalg::BaseTableOp>(
        loc,
        tupleStreamType,
        builder->getStringAttr("test_table"),
        builder->getI64IntegerAttr(12346)
    );
    
    // Create MaterializeOp with all columns
    auto columnsAttr = builder->getArrayAttr({
        builder->getStringAttr("id"),
        builder->getStringAttr("name"),
        builder->getStringAttr("price")
    });
    auto materializeOp = builder->create<mlir::relalg::MaterializeOp>(
        loc,
        tableType,
        baseTableOp.getResult(),
        columnsAttr
    );
    
    builder->create<func::ReturnOp>(loc, materializeOp.getResult());
    
    // Run the pass
    PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(module);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Count DSAppend operations - should be 3 (one per column)
    int appendCount = 0;
    funcOp.walk([&](mlir::dsa::Append op) {
        appendCount++;
    });
    
    EXPECT_EQ(appendCount, 3) << "Should generate one ds_append per column";
    
    PGX_DEBUG("Multiple column test passed");
}