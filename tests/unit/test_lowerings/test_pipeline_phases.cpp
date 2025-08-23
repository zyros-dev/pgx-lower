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
    const char* simpleMLIR = R"(
        module {
            func.func @main() -> i32 {
                %c42 = arith.constant 42 : i32
                return %c42 : i32
            }
        }
    )";
    
    ASSERT_TRUE(tester->loadRelAlgModule(simpleMLIR)) 
        << "Failed to load MLIR module: " << tester->getLastError();
    
    EXPECT_TRUE(tester->runCompletePipeline()) 
        << "Complete pipeline error: " << tester->getLastError();
    
    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";
    
    // Check that we got LLVM operations in the final result
    std::string finalMLIR = tester->getCurrentMLIR();
    EXPECT_TRUE(finalMLIR.find("llvm.") != std::string::npos) 
        << "Expected LLVM dialect operations in final MLIR";
}
