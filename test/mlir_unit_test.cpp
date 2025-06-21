#include <gtest/gtest.h>
#include <llvm/Config/llvm-config.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <src/pgx_lower/src/mlir_runner.h>
#include <fstream>
#include <cstdio>
#include <unistd.h>

TEST(MLIRTest, PassManagerSetup) {
    EXPECT_GT(LLVM_VERSION_MAJOR, 0);
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    mlir::PassManager pm(&context);
    EXPECT_NO_THROW(pm.addPass(mlir::createCanonicalizerPass()));
    EXPECT_NO_THROW(pm.addPass(mlir::createConvertFuncToLLVMPass()));
    EXPECT_NO_THROW(pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass()));
}

TEST(MLIRTest, BuildAndRunMlir) {
    auto result = mlir_runner::run_mlir_test(5);
    EXPECT_TRUE(result) << "Expected failure at JIT execution due to incomplete lowering - this is expected for now";
}

TEST(MLIRTest, ExternalFunctionCall) {
    // Create a temporary file and write the number 6 to it
    std::string tempFilename = "/tmp/mlir_test_" + std::to_string(getpid()) + ".txt";
    std::ofstream outFile(tempFilename);
    ASSERT_TRUE(outFile.is_open()) << "Failed to create temporary file";
    outFile << "6" << std::endl;
    outFile.close();
    
    // Create an external function that reads from the file
    auto fileReader = [tempFilename]() -> int64_t {
        std::ifstream inFile(tempFilename);
        if (!inFile.is_open()) {
            return -1; // Error reading file
        }
        int64_t value;
        inFile >> value;
        inFile.close();
        return value;
    };
    
    // Test the MLIR JIT with external function call
    // This should read 6 from file and add 10 (from run_external_func_test), resulting in 16
    auto result = mlir_runner::run_external_func_test(fileReader);
    EXPECT_TRUE(result) << "MLIR JIT with external function call should succeed";
    
    // Clean up temporary file
    std::remove(tempFilename.c_str());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 