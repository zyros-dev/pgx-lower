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
#include <iostream>

namespace {

class MaterializeOnlyTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    MaterializeOnlyTest() : builder(&context) {
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
    
    ~MaterializeOnlyTest() {
        std::cerr << "[TEST] MaterializeOnlyTest destructor called\n";
        if (module) {
            std::cerr << "[TEST] Destroying module\n";
            module->destroy();
            std::cerr << "[TEST] Module destroyed\n";
        }
        std::cerr << "[TEST] MaterializeOnlyTest destructor completed\n";
    }
};

TEST_F(MaterializeOnlyTest, MaterializeWithoutReturn) {
    std::cerr << "[TEST] Creating function without return\n";
    
    // Create function with void return type (no returns)
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "materialize_test", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    std::cerr << "[TEST] Creating BaseTableOp\n";
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    std::cerr << "[TEST] Creating MaterializeOp\n";
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    // Don't use the result, just have the MaterializeOp there
    
    std::cerr << "[TEST] Creating empty ReturnOp\n";
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    
    std::cerr << "[TEST] Running RelAlgToDB pass\n";
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    std::cerr << "[TEST] Calling pm.run()\n";
    auto result = pm.run(funcOp);
    std::cerr << "[TEST] pm.run() returned\n";
    
    std::cerr << "[TEST] Checking pass result\n";
    if (succeeded(result)) {
        std::cerr << "[TEST] Pass succeeded, counting operations\n";
        
        // Count operations
        int dbOpCount = 0;
        int dsaOpCount = 0;
        int relalgOpCount = 0;
        
        std::cerr << "[TEST] Starting walk\n";
        funcOp.walk([&](mlir::Operation* op) {
            auto dialectName = op->getName().getDialectNamespace();
            if (dialectName == "db") dbOpCount++;
            else if (dialectName == "dsa") dsaOpCount++;
            else if (dialectName == "relalg") relalgOpCount++;
        });
        
        std::cerr << "[TEST] Walk completed\n";
        std::cerr << "[TEST] DB ops: " << dbOpCount << ", DSA ops: " << dsaOpCount 
                  << ", RelAlg ops remaining: " << relalgOpCount << "\n";
        
        EXPECT_GT(dbOpCount, 0) << "Should generate DB operations for PostgreSQL table access (Phase 4d)";
        EXPECT_GT(dsaOpCount, 0) << "Should generate DSA operations for result building";
        EXPECT_EQ(relalgOpCount, 0) << "All RelAlg operations should be erased";
        
        std::cerr << "[TEST] Test assertions completed\n";
        SUCCEED() << "Pass ran successfully without segfault";
    } else {
        std::cerr << "[TEST] Pass failed\n";
        FAIL() << "Pass failed";
    }
    std::cerr << "[TEST] Test function returning\n";
}

} // namespace