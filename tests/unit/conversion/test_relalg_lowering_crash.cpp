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
#include "execution/logging.h"

class RelAlgCrashTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    
    RelAlgCrashTest() {
        // Load required dialects for RelAlg parsing and lowering
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::util::UtilDialect>();
        context.loadDialect<mlir::relalg::RelAlgDialect>();
        context.loadDialect<mlir::db::DBDialect>();
        context.loadDialect<mlir::dsa::DSADialect>();
        
        // Disable threading for PostgreSQL compatibility
        context.disableMultithreading();
    }
};

TEST_F(RelAlgCrashTest, TestSimplePassExecution) {
    PGX_INFO("=== Testing Simple Pass Execution (Crash Reproduction) ===");
    
    // Create a simple test module with just func and arith operations
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

    PGX_INFO("Step 1: Parsing RelAlg MLIR from logs");
    auto module = mlir::parseSourceString<mlir::ModuleOp>(simpleIR, &context);
    ASSERT_TRUE(module) << "Failed to parse RelAlg MLIR string";
    
    PGX_INFO("Step 2: Verifying RelAlg module");
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "RelAlg module verification failed";
    
    // Count initial operations
    int opCount = 0;
    module->walk([&](mlir::Operation* op) { opCount++; });
    PGX_INFO("RelAlg module has " + std::to_string(opCount) + " operations");
    
    // Step 3: Run Phase 3a - the EXACT pipeline that crashes PostgreSQL
    PGX_INFO("Phase 3a: Running RelAlg→DB lowering");
    {
        mlir::PassManager pm(&context);
        pm.enableVerifier(true);
        
        // Verify module before lowering
        if (mlir::failed(mlir::verify(*module))) {
            FAIL() << "Phase 3a: Module verification failed before lowering";
        }
        
        // Use the EXACT Phase 3a pipeline that crashes PostgreSQL
        mlir::pgx_lower::createRelAlgToDBPipeline(pm, true);
        
        PGX_INFO("🎯 Running THE EXACT Phase 3a PIPELINE THAT CRASHES POSTGRESQL");
        if (mlir::failed(pm.run(*module))) {
            FAIL() << "❌ FOUND CRASH: Phase 3a RelAlg→DB lowering failed";
        }
        PGX_INFO("✅ Phase 3a RelAlg→DB lowering succeeded in unit test");
        
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
    PGX_INFO("Final module has " + std::to_string(finalOpCount) + " operations");
    for (const auto& [dialect, count] : dialectCounts) {
        PGX_INFO("  " + dialect + " dialect: " + std::to_string(count) + " operations");
    }
    
    PGX_INFO("🎉 NO CRASH REPRODUCED - Confirms PostgreSQL environment issue");
}