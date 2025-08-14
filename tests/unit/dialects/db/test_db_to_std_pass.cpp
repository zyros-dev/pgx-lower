#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "execution/logging.h"

class DBToStdPassTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    
    void SetUp() override {
        // Load required dialects
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
    }
};

TEST_F(DBToStdPassTest, EmptyModulePass) {
    // Test that the pass works on an empty module
    const char* moduleStr = R"mlir(
        module {
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
    ASSERT_TRUE(module) << "Failed to parse module";
    
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    auto result = pm.run(*module);
    EXPECT_TRUE(mlir::succeeded(result)) << "Pass should succeed on empty module";
}

TEST_F(DBToStdPassTest, SimpleTableScanOp) {
    // Test lowering a simple TableScanOp
    const char* moduleStr = R"mlir(
        module {
            func.func @test() -> !db.runtime_call {
                %0 = db.table_scan "test" 4676466 : !db.runtime_call
                return %0 : !db.runtime_call
            }
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
    ASSERT_TRUE(module) << "Failed to parse module";
    
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    auto result = pm.run(*module);
    EXPECT_TRUE(mlir::succeeded(result)) << "Pass should succeed on TableScanOp";
    
    // Verify TableScanOp was lowered
    bool hasTableScanOp = false;
    module->walk([&](mlir::db::TableScanOp op) {
        hasTableScanOp = true;
    });
    EXPECT_FALSE(hasTableScanOp) << "TableScanOp should be lowered";
}

TEST_F(DBToStdPassTest, MaterializeOp) {
    // Test lowering a MaterializeOp
    const char* moduleStr = R"mlir(
        module {
            func.func @test(%runtime : !db.runtime_call) -> !util.tuple {
                %0 = db.materialize %runtime : (!db.runtime_call) -> !util.tuple
                return %0 : !util.tuple
            }
        }
    )mlir";
    
    auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
    ASSERT_TRUE(module) << "Failed to parse module";
    
    mlir::PassManager pm(&context);
    pm.addPass(mlir::db::createLowerToStdPass());
    
    auto result = pm.run(*module);
    EXPECT_TRUE(mlir::succeeded(result)) << "Pass should succeed on MaterializeOp";
    
    // Verify MaterializeOp was lowered
    bool hasMaterializeOp = false;
    module->walk([&](mlir::db::MaterializeOp op) {
        hasMaterializeOp = true;
    });
    EXPECT_FALSE(hasMaterializeOp) << "MaterializeOp should be lowered";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}