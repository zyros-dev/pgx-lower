#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/util/UtilDialect.h>
#include <mlir/Transforms/Passes.h>
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Passes.h"

class RelAlgCrashTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    
    RelAlgCrashTest() {
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::relalg::RelAlgDialect>();
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        
        context.disableMultithreading();
    }
};

TEST_F(RelAlgCrashTest, TestSimplePassExecution) {
    
    const char* simpleIR = R"(
module {
  func.func @main() {
    %0 = relalg.basetable  {table_identifier = "test|oid:16384"} columns: {col_2 => @test::@col_2({type = i32}), col_3 => @test::@col_3({type = i32})}
    %1 = relalg.map %0 computes : [@map::@addition({type = i32})] (%arg0: !relalg.tuple){
      %3 = relalg.getcol %arg0 @test::@col_2 : i32
      %4 = relalg.getcol %arg0 @test::@col_3 : i32
      %5 = arith.addi %3, %4 : i32
      relalg.return %5 : i32
    }
    %2 = relalg.materialize %1 [@map::@addition] => ["addition"] : !dsa.table
    return
  }
}

)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(simpleIR, &context);
    ASSERT_TRUE(module) << "Failed to parse RelAlg MLIR string";
    
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "RelAlg module verification failed";
    
    int opCount = 0;
    module->walk([&](mlir::Operation* op) { opCount++; });
    
    {
        mlir::PassManager pm(&context);
        pm.enableVerifier(true);
        
        if (mlir::failed(mlir::verify(*module))) {
            FAIL() << "Phase 3a: Module verification failed before lowering";
        }
        
        mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);
        
        if (mlir::failed(pm.run(*module))) {
            FAIL() << "❌ FOUND CRASH: Phase 3a RelAlg→DB lowering failed";
        }
        
        // Verify module after lowering
        if (mlir::failed(mlir::verify(*module))) {
            FAIL() << "Phase 3a: Module verification failed after lowering";
        }
    }
    
    // Count final operations after complete lowering
    int finalOpCount = 0;
    std::map<std::string, int> dialectCounts;
    module->walk([&](mlir::Operation* op) { 
        finalOpCount++; 
        std::string dialectName = op->getDialect()->getNamespace().str();
        dialectCounts[dialectName]++;
    });
    for (const auto& [dialect, count] : dialectCounts) {
    }
    
}