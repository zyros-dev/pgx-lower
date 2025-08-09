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

//===----------------------------------------------------------------------===//
// Phase 4d: Basic Pipeline Test  
// Validates core streaming architecture with mixed DB+DSA operations
//===----------------------------------------------------------------------===//

namespace {

class Phase4c4BasicTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    Phase4c4BasicTest() : builder(&context) {
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
    
    ~Phase4c4BasicTest() {
        if (module) module->destroy();
    }
};

TEST_F(Phase4c4BasicTest, PassRunsSuccessfully) {
    // Create the Test 1 pipeline
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {relAlgTableType});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "test1_query", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create pipeline: BaseTable -> Materialize -> Return
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), materializeOp.getResult());
    
    // Run the RelAlgToDB pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass should run successfully";
    
    // Basic verification - count operation types
    int dbOpCount = 0;
    int dsaOpCount = 0;
    int relalgOpCount = 0;
    
    funcOp.walk([&](mlir::Operation* op) {
        auto dialectName = op->getName().getDialectNamespace();
        if (dialectName == "db") dbOpCount++;
        else if (dialectName == "dsa") dsaOpCount++;
        else if (dialectName == "relalg") relalgOpCount++;
    });
    
    // Verify transformation occurred (Mixed DB+DSA pattern for PostgreSQL)
    EXPECT_GT(dbOpCount, 0) << "Should generate DB operations for PostgreSQL table access";
    EXPECT_GT(dsaOpCount, 0) << "Should generate DSA operations for result materialization";
    EXPECT_EQ(relalgOpCount, 0) << "All RelAlg operations should be erased";
    
    std::cerr << "Phase 4d Basic Test Results:\n";
    std::cerr << "  DB operations generated: " << dbOpCount << "\n";
    std::cerr << "  DSA operations generated: " << dsaOpCount << "\n";
    std::cerr << "  RelAlg operations remaining: " << relalgOpCount << "\n";
}

TEST_F(Phase4c4BasicTest, ProducerConsumerPattern) {
    // Test that the producer-consumer pattern is working
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {relAlgTableType});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "producer_consumer_test", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create pipeline
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("id")}));
    
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), materializeOp.getResult());
    
    // Run pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "Pass should run successfully";
    
    // Check for mixed DB+DSA pattern (PostgreSQL integration)
    bool hasGetExternal = false;
    bool hasIterateExternal = false;
    bool hasWhileLoop = false;
    bool hasDSAppend = false;
    int dbOpCount = 0;
    int dsaOpCount = 0;
    
    funcOp.walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "db.get_external") hasGetExternal = true;
        if (op->getName().getStringRef() == "db.iterate_external") hasIterateExternal = true;
        if (op->getName().getStringRef() == "scf.while") hasWhileLoop = true;
        if (op->getName().getStringRef() == "dsa.ds_append") hasDSAppend = true;
        if (op->getDialect() && op->getDialect()->getNamespace() == "db") dbOpCount++;
        if (op->getDialect() && op->getDialect()->getNamespace() == "dsa") dsaOpCount++;
    });
    
    EXPECT_TRUE(hasGetExternal) << "Should have db.get_external for PostgreSQL table access";
    EXPECT_TRUE(hasIterateExternal) << "Should have db.iterate_external for tuple iteration";
    EXPECT_TRUE(hasWhileLoop) << "Should have scf.while loop for iteration";
    EXPECT_TRUE(hasDSAppend) << "Should have dsa.ds_append for materialization";
    EXPECT_GT(dbOpCount, 0) << "Should have DB operations for PostgreSQL integration";
    EXPECT_GT(dsaOpCount, 0) << "Should have DSA operations for result materialization";
}

} // namespace