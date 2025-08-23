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

TEST_F(MLIRLoweringPipelineTest, BasicArithmeticPipeline) {
    // Test basic arithmetic pipeline - equivalent to Test 9 which currently fails
    const char* basicArithmeticMLIR = R"(
        module {
            func.func @main() -> i32 {
                %c1 = arith.constant 1 : i32
                %c2 = arith.constant 2 : i32
                %sum = arith.addi %c1, %c2 : i32
                return %sum : i32
            }
        }
    )";
    
    std::cout << "=== Testing Basic Arithmetic Pipeline ===" << std::endl;
    ASSERT_TRUE(tester->loadRelAlgModule(basicArithmeticMLIR)) 
        << "Failed to load MLIR module: " << tester->getLastError();
    
    std::cout << "Phase 3a: RelAlg -> DB+DSA+Util..." << std::endl;
    EXPECT_TRUE(tester->runPhase3a()) << "Phase 3a error: " << tester->getLastError();
    EXPECT_TRUE(tester->isPhase3aComplete());
    
    std::cout << "Phase 3b: DB+DSA+Util -> Standard..." << std::endl;
    EXPECT_TRUE(tester->runPhase3b()) << "Phase 3b error: " << tester->getLastError();
    EXPECT_TRUE(tester->isPhase3bComplete());

    std::string mlir = tester->getCurrentMLIR();
    EXPECT_TRUE(mlir.find("func.") != std::string::npos || 
                mlir.find("arith.") != std::string::npos ||
                mlir.find("scf.") != std::string::npos) << "Expected Standard MLIR operations";
    
    std::cout << "Phase 3c: Standard -> LLVM..." << std::endl;
    EXPECT_TRUE(tester->runPhase3c()) << "Phase 3c error: " << tester->getLastError();
    EXPECT_TRUE(tester->isPhase3cComplete());
    
    std::string llvmIR = tester->getLLVMIR();
    EXPECT_TRUE(llvmIR.find("define") != std::string::npos) << "Expected LLVM IR with function definitions";
    
    EXPECT_TRUE(tester->verifyCurrentModule()) << "Final module should be valid";
    std::cout << "=== Pipeline Test Complete ===" << std::endl;
}

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
