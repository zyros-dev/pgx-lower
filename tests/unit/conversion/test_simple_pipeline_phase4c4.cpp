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
#include <sstream>

//===----------------------------------------------------------------------===//
// Phase 4c-4: Simple Pipeline Test - Debugging segfault
//===----------------------------------------------------------------------===//

namespace {

class SimplePipelineTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    
    SimplePipelineTest() : builder(&context) {
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
    
    ~SimplePipelineTest() {
        if (module) module->destroy();
    }
};

TEST_F(SimplePipelineTest, SimpleTranslation) {
    // Create the simplest possible pipeline
    
    // Create function type
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {relAlgTableType});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "simple_query", funcType);
    
    // Create function body
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    // Create MaterializeOp
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    // Create func::ReturnOp
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), materializeOp.getResult());
    
    // Print IR before pass
    std::cerr << "\n=== IR Before Pass ===\n";
    funcOp.print(llvm::errs());
    std::cerr << "\n";
    
    // Run the RelAlgToDB pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Verify the function
    auto verifyResult = funcOp.verify();
    ASSERT_TRUE(succeeded(verifyResult)) << "Function verification failed after pass";
    
    // Now try to print the IR
    std::cerr << "\n=== IR After Pass ===\n";
    funcOp.print(llvm::errs());
    std::cerr << "\n";
}

TEST_F(SimplePipelineTest, ManualTranslationCheck) {
    // Manually check what the translation produces
    
    auto relAlgTableType = pgx::mlir::relalg::TableType::get(&context);
    auto funcType = builder.getFunctionType({}, {relAlgTableType});
    auto funcOp = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "manual_check", funcType);
    
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create operations
    auto baseTableOp = builder.create<pgx::mlir::relalg::BaseTableOp>(
        builder.getUnknownLoc(), builder.getStringAttr("test"), 12345);
    
    auto materializeOp = builder.create<pgx::mlir::relalg::MaterializeOp>(
        builder.getUnknownLoc(), relAlgTableType, baseTableOp.getResult(), 
        builder.getArrayAttr({builder.getStringAttr("*")}));
    
    auto returnOp = builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), materializeOp.getResult());
    
    // Check initial state
    EXPECT_EQ(returnOp.getNumOperands(), 1);
    EXPECT_EQ(returnOp.getOperand(0), materializeOp.getResult());
    
    // Run the pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::pgx_conversion::createRelAlgToDBPass());
    
    auto result = pm.run(funcOp);
    ASSERT_TRUE(succeeded(result)) << "RelAlgToDB pass failed";
    
    // Walk the function to see what operations remain
    std::cerr << "\n=== Operations after pass ===\n";
    funcOp.walk([&](mlir::Operation* op) {
        std::cerr << "  - " << op->getName().getStringRef().str() << "\n";
    });
}

} // namespace