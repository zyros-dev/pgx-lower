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

class SegfaultDebugTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    SegfaultDebugTest() : builder(&context) {
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
    
    ~SegfaultDebugTest() {
        if (module) module->destroy();
    }
};

TEST_F(SegfaultDebugTest, MinimalPipelineWithLogging) {
    std::cerr << "[TEST] Creating function\n";
    
    // Create function with RelAlg table return type
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {relAlgTableType});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "minimal_test", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    std::cerr << "[TEST] Creating BaseTableOp\n";
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    std::cerr << "[TEST] Creating MaterializeOp\n";
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    std::cerr << "[TEST] Creating ReturnOp\n";
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), materializeOp.getResult());
    
    std::cerr << "[TEST] Function created, skipping dump to avoid segfault\n";
    // funcOp.dump();  // SKIP - this might be causing segfault
    
    std::cerr << "[TEST] Running RelAlgToDB pass\n";
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    
    if (mlir::succeeded(result)) {
        std::cerr << "[TEST] Pass succeeded\n";
        
        // Verify operations were transformed
        int dbOpCount = 0;
        int dsaOpCount = 0;
        int relalgOpCount = 0;
        
        funcOp.walk([&](mlir::Operation* op) {
            auto dialectName = op->getName().getDialectNamespace();
            if (dialectName == "db") dbOpCount++;
            else if (dialectName == "dsa") dsaOpCount++;
            else if (dialectName == "relalg") relalgOpCount++;
        });
        
        std::cerr << "[TEST] DB ops: " << dbOpCount << ", DSA ops: " << dsaOpCount 
                  << ", RelAlg ops remaining: " << relalgOpCount << "\n";
        
        EXPECT_GT(dbOpCount, 0) << "Should generate DB operations for table access (Phase 4d)";
        EXPECT_GT(dsaOpCount, 0) << "Should generate DSA operations for result building";
        EXPECT_EQ(relalgOpCount, 0) << "All RelAlg operations should be erased";
        
        // Try to check function type without dumping
        auto newFuncType = funcOp.getFunctionType();
        std::cerr << "[TEST] Function has " << newFuncType.getNumResults() << " results\n";
        
        SUCCEED() << "Pass ran successfully";
    } else {
        FAIL() << "Pass failed";
    }
}

} // namespace