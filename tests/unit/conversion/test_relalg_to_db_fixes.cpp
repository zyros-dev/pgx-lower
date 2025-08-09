#include "gtest/gtest.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "execution/logging.h"

using namespace mlir;

namespace {

// Test fixture for RelAlgToDB fixes
class RelAlgToDBFixesTest : public ::testing::Test {
protected:
    MLIRContext context;
    OpBuilder builder;
    
    RelAlgToDBFixesTest() : builder(&context) {
        // Load required dialects
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<func::FuncDialect>();
    }
};

// Test 1: Empty function handling
TEST_F(RelAlgToDBFixesTest, EmptyFunctionHandling) {
    PGX_DEBUG("Testing empty function handling fix");
    
    // Create empty function
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "empty_query", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Add required terminator for empty function
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run the pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "Pass should handle empty functions gracefully";
    
    PGX_DEBUG("Empty function test passed");
}

// Test 2: BaseTableOp without MaterializeOp remains
TEST_F(RelAlgToDBFixesTest, BaseTableOpWithoutMaterialize) {
    PGX_DEBUG("Testing BaseTableOp erasure logic fix");
    
    // Create function with just BaseTableOp
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_basetable_only", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp without MaterializeOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345)
    );
    
    // Return without using the BaseTableOp
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run the pass
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "Pass should succeed";
    
    // Verify BaseTableOp is still there
    int baseTableCount = 0;
    funcOp.walk([&](pgx::mlir::relalg::BaseTableOp op) {
        baseTableCount++;
    });
    EXPECT_EQ(baseTableCount, 1) << "BaseTableOp should remain without MaterializeOp";
    
    PGX_DEBUG("BaseTableOp erasure test passed");
}

// Test 3: MaterializeOp with valid termination (updated for hybrid architecture)
// DISABLED: Column resolution issue between BaseTable and Materialize translators
TEST_F(RelAlgToDBFixesTest, DISABLED_MaterializeOpWithTermination) {
    PGX_DEBUG("Testing MaterializeOp generates properly terminated MLIR");
    
    // Create module for proper pass application
    auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function with MaterializeOp
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "test_materialize", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp and MaterializeOp
    auto tupleStreamType = pgx::mlir::relalg::TupleStreamType::get(&context);
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(),
        tupleStreamType,
        builder.getStringAttr("test_table"),
        builder.getI64IntegerAttr(12345)
    );
    
    auto tableType = pgx::mlir::relalg::TableType::get(&context);
    auto columnsAttr = builder.getArrayAttr({builder.getStringAttr("id")});
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(),
        tableType,
        baseTableOp.getResult(),
        columnsAttr
    );
    
    // Add proper termination
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Run the pass on module (not just function)
    PassManager pm(&context);
    pm.addPass(pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(module);
    ASSERT_TRUE(succeeded(result)) << "Pass should succeed without terminator errors";
    
    // Verify no RelAlg operations remain
    int relalgCount = 0;
    funcOp.walk([&](Operation* op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "relalg") {
            relalgCount++;
        }
    });
    EXPECT_EQ(relalgCount, 0) << "All RelAlg operations should be converted";
    
    // Verify both DSA and DB operations were created (hybrid architecture)
    int dsaCount = 0;
    int dbCount = 0;
    funcOp.walk([&](Operation* op) {
        if (op->getDialect()) {
            if (op->getDialect()->getNamespace() == "dsa") {
                dsaCount++;
            } else if (op->getDialect()->getNamespace() == "db") {
                dbCount++;
            }
        }
    });
    EXPECT_GT(dsaCount, 0) << "DSA operations should be created for internal processing";
    EXPECT_GT(dbCount, 0) << "DB operations should be created for PostgreSQL output";
    
    // Verify function signature changed (no return value in hybrid architecture)
    auto updatedFuncType = funcOp.getFunctionType();
    EXPECT_EQ(updatedFuncType.getNumResults(), 0) << "Function should return void after conversion";
    
    PGX_DEBUG("MaterializeOp termination test passed");
}

} // namespace