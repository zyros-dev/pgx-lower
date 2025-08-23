#include <gtest/gtest.h>
#include "standalone_mlir_runner.h"

using namespace pgx_test;

class MLIRLoweringPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        tester = std::make_unique<StandalonePipelineTester>();
    }
    
    void TearDown() override {
        tester.reset();
    }
    
    std::unique_ptr<StandalonePipelineTester> tester;
};

TEST_F(MLIRLoweringPipelineTest, SimpleConstantPipeline) {
    // Very basic test - just constant return
    auto simpleMLIR = R"(
        module {
            func.func @main() -> i32 {
                %c42 = arith.constant 42 : i32
                return %c42 : i32
            }
        }
    )";
    
    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) 
        << "Failed to load MLIR module";
    
    EXPECT_TRUE(tester->runPhase3a() && tester->runPhase3b() && tester->runPhase3c())
        << "Complete pipeline failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";
    
    // Check that we got LLVM operations in the final result
    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) 
        << "Expected LLVM dialect operations in final MLIR";
}

TEST_F(MLIRLoweringPipelineTest, Test9Arith) {
    // Very basic test - just constant return
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {table_identifier = "test_arithmetic|oid:8338330"} columns: {id => @test_arithmetic::@id({type = i32}), val1 => @test_arithmetic::@val1({type = i32}), val2 => @test_arithmetic::@val2({type = i32})}
            %1 = relalg.map %0 computes : [@map::@addition({type = i32})] (%arg0: !relalg.tuple){
              %3 = relalg.getcol %arg0 @test_arithmetic::@val1 : i32
              %4 = relalg.getcol %arg0 @test_arithmetic::@val2 : i32
              %5 = db.add %3 : i32, %4 : i32
              relalg.return %5 : i32
            }
            %2 = relalg.materialize %1 [@map::@addition] => ["addition"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a failed";
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b failed";
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";
}

TEST_F(MLIRLoweringPipelineTest, Test11) {
    // This replicates the relalg from test 11. If you want to change this input, you need to change the AST parser
    // to produce that as well.
    auto simpleMLIR = R"(
        module {
          func.func @main() {
            %0 = relalg.basetable  {table_identifier = "test_logical|oid:8493992"} columns: {flag1 => @test_logical::@flag1({type = i1}), flag2 => @test_logical::@flag2({type = i1}), id => @test_logical::@id({type = i32}), value => @test_logical::@value({type = i32})}
            %1 = relalg.map %0 computes : [@map::@and_result({type = i32})] (%arg0: !relalg.tuple){
              %3 = relalg.getcol %arg0 @test_logical::@flag1 : i1
              %4 = relalg.getcol %arg0 @test_logical::@flag2 : i1
              %5 = db.and %3, %4 : i1, i1
              relalg.return %5 : i1
            }
            %2 = relalg.materialize %1 [@map::@and_result] => ["and_result"] : !dsa.table
            return
          }
        }
    )";

    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) << "Failed to load MLIR module";

    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a failed";
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b failed";
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c failed";

    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";

    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) << "Expected LLVM dialect operations in final MLIR";
}
