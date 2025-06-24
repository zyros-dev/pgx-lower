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
#include <core/mlir_runner.h>
#include "../test_helpers.h"

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

    mlir::PassManager pm(&context);
    EXPECT_NO_THROW(pm.addPass(mlir::createCanonicalizerPass()));
    EXPECT_NO_THROW(pm.addPass(mlir::createConvertFuncToLLVMPass()));
    EXPECT_NO_THROW(pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass()));
}

TEST_F(MLIRRunnerTest, PostgreSQLTableScanInMLIR) {
    std::vector<int64_t> mockData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    MockTupleScanContext mockContext = {mockData, 0, true};
    g_mock_scan_context = &mockContext;
    
    ConsoleLogger logger;
    
    auto result = mlir_runner::run_mlir_postgres_table_scan("test_table", logger);
    
    EXPECT_TRUE(result) << "MLIR PostgreSQL table scan should succeed";
}

TEST_F(MLIRRunnerTest, PostgreSQLTypedFieldAccess) {
    
    // Set up mock data for the test
    std::vector<int64_t> mockData = {100, 200, 300};
    MockTupleScanContext mockContext = {mockData, 0, true};
    g_mock_scan_context = &mockContext;
    
    // Test typed field access with pg dialect
    ConsoleLogger logger;
    
    // This should generate MLIR with pg.scan_table, pg.read_tuple, pg.get_int_field, pg.get_text_field
    // and then show the transformation via lowering pass
    auto result = mlir_runner::run_mlir_postgres_typed_table_scan("test_table", logger);
    
    EXPECT_TRUE(result) << "MLIR PostgreSQL typed field access should succeed";
    
    std::cout << "[TEST] PostgreSQL typed field access completed!" << std::endl;
}