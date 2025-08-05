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
#include <execution/mlir_runner.h>
#include "../test_helpers.h"
#include <signal.h>
#include <compiler/Dialect/RelAlg/RelAlgDialect.h>
#include <compiler/Dialect/RelAlg/RelAlgOps.h>
#include <compiler/Dialect/DB/DBDialect.h>
#include <compiler/Dialect/DSA/DSADialect.h>
#include <compiler/Dialect/util/UtilDialect.h>
#include <compiler/Dialect/TupleStream/TupleStreamDialect.h>
#include <compiler/Dialect/TupleStream/TupleStreamOps.h>
#include <runtime/helpers.h>

// DataSource_get stub for unit tests
extern "C" void* DataSource_get(pgx_lower::compiler::runtime::VarLen32 description) {
    // Mock implementation for unit tests
    static int mock_datasource = 42;
    return &mock_datasource;
}

class MLIRRunnerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup for each test
    }

    void TearDown() override {
        // Cleanup mock context
        g_mock_scan_context = nullptr;
    }
};

TEST_F(MLIRRunnerTest, PassManagerSetup) {
    EXPECT_GT(LLVM_VERSION_MAJOR, 0);
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    auto pm = mlir::PassManager(&context);
    EXPECT_NO_THROW(pm.addPass(mlir::createCanonicalizerPass()));
    EXPECT_NO_THROW(pm.addPass(mlir::createConvertFuncToLLVMPass()));
    EXPECT_NO_THROW(pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass()));
}


