// Test for verifying MaterializeTranslator consumer chain execution
// This test ensures MaterializeTranslator::consume() is called and generates operations

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

class ConsumerChainTest : public ::testing::Test {
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

// Test that MaterializeTranslator::consume() generates Append and StoreResultOp
TEST_F(ConsumerChainTest, DISABLED_ConsumerChainExecutes) {
    PGX_DEBUG("Testing MaterializeTranslator consumer chain execution");
    
    // Create module and function
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToStart(module.getBody());
    
    // Function starts with RelAlg table type
    auto tableType = mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_consumer_chain", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp with specific table name
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<mlir::relalg::BaseTableOp>(
        loc,
        tupleStreamType,
        builder->getStringAttr("consumer_test_table"),
        builder->getI64IntegerAttr(99999)  // table_oid
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
    
    // Run the RelAlgToDB pass on the module
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(mlir::pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(module);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Debug: Print the module after pass
    PGX_DEBUG("Generated MLIR after RelAlgToDB pass:");
    module.print(llvm::errs());
    llvm::errs() << "\n";
    
    // Verify consumer chain operations were generated
    bool foundDSAppend = false;
    bool foundStoreResult = false;
    bool foundNextRow = false;
    int dsAppendCount = 0;
    int storeResultCount = 0;
    
    module.walk([&](Operation *op) {
        if (isa<mlir::dsa::Append>(op)) {
            foundDSAppend = true;
            dsAppendCount++;
            PGX_DEBUG("Found Append - consumer chain executed!");
        } else if (isa<pgx::db::StoreResultOp>(op)) {
            foundStoreResult = true;
            storeResultCount++;
            PGX_DEBUG("Found StoreResultOp - consumer chain executed!");
        } else if (isa<mlir::dsa::NextRow>(op)) {
            foundNextRow = true;
            PGX_DEBUG("Found NextRow - consumer chain executed!");
        }
    });
    
    // Verify consumer chain executed
    EXPECT_TRUE(foundDSAppend) << "MaterializeTranslator::consume() should generate dsa.ds_append";
    EXPECT_TRUE(foundStoreResult) << "MaterializeTranslator::consume() should generate db.store_result";
    EXPECT_TRUE(foundNextRow) << "MaterializeTranslator::consume() should generate dsa.next_row";
    
    // Should have one append and store per column (1 for "id")
    EXPECT_EQ(dsAppendCount, 1) << "Should have one dsa.ds_append for 'id' column";
    EXPECT_EQ(storeResultCount, 1) << "Should have one db.store_result for 'id' column";
    
    PGX_DEBUG("Consumer chain test passed - MaterializeTranslator::consume() is being called!");
}

// Test that consumer chain works inside the scf.while loop
TEST_F(ConsumerChainTest, ConsumerChainInLoop) {
    PGX_DEBUG("Testing consumer chain operations are inside the loop");
    
    auto loc = builder->getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToStart(module.getBody());
    
    auto tableType = mlir::relalg::TableType::get(&context);
    auto funcType = builder->getFunctionType({}, {tableType});
    auto funcOp = builder->create<func::FuncOp>(loc, "test_loop_consumer", funcType);
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create simple BaseTable -> Materialize pipeline
    auto tupleStreamType = mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder->create<mlir::relalg::BaseTableOp>(
        loc,
        tupleStreamType,
        builder->getStringAttr("loop_test_table"),
        builder->getI64IntegerAttr(88888)
    );
    
    auto columnsAttr = builder->getArrayAttr({builder->getStringAttr("id")});
    auto materializeOp = builder->create<mlir::relalg::MaterializeOp>(
        loc,
        tableType,
        baseTableOp.getResult(),
        columnsAttr
    );
    
    builder->create<func::ReturnOp>(loc, materializeOp.getResult());
    
    // Run the pass
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(mlir::pgx_conversion::createRelAlgToDBPass());
    
    LogicalResult result = pm.run(module);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify operations are inside the while loop
    bool foundWhileOp = false;
    bool operationsInLoop = false;
    
    module.walk([&](scf::WhileOp whileOp) {
        foundWhileOp = true;
        
        // Check if Append and StoreResultOp are inside the loop
        whileOp.getAfter().walk([&](Operation *op) {
            if (isa<mlir::dsa::Append>(op) || 
                isa<pgx::db::StoreResultOp>(op)) {
                operationsInLoop = true;
            }
        });
    });
    
    EXPECT_TRUE(foundWhileOp) << "Should generate scf.while loop for iteration";
    EXPECT_TRUE(operationsInLoop) << "Consumer operations should be inside the loop";
    
    PGX_DEBUG("Consumer chain correctly placed inside iteration loop");
}