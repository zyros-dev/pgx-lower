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
#include <vector>

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

#ifndef POSTGRESQL_EXTENSION
struct MockTupleScanContext {
    std::vector<int64_t> values;
    size_t currentIndex;
    bool hasMore;
};

static MockTupleScanContext* g_mock_scan_context = nullptr;

extern "C" int64_t mock_get_next_tuple() {
    if (!g_mock_scan_context) {
        return -1;
    }
    
    if (g_mock_scan_context->currentIndex >= g_mock_scan_context->values.size()) {
        g_mock_scan_context->hasMore = false;
        return -2;
    }
    
    int64_t value = g_mock_scan_context->values[g_mock_scan_context->currentIndex];
    g_mock_scan_context->currentIndex++;
    g_mock_scan_context->hasMore = true;
    
    return value;
}

extern "C" void* open_postgres_table(const char* tableName) {
    if (!g_mock_scan_context) {
        return nullptr;
    }
    return g_mock_scan_context;
}

extern "C" int64_t read_next_tuple_from_table(void* tableHandle) {
    if (!tableHandle) {
        return -1;
    }
    
    MockTupleScanContext* context = static_cast<MockTupleScanContext*>(tableHandle);
    return mock_get_next_tuple();
}

extern "C" void close_postgres_table(void* tableHandle) {
    // Nothing to do for mock implementation
}

extern "C" bool add_tuple_to_result(int64_t value) {
    return true;
}


TEST(MLIRTest, PostgreSQLTableScanInMLIR) {
    std::vector<int64_t> mockData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    MockTupleScanContext mockContext = {mockData, 0, true};
    g_mock_scan_context = &mockContext;
    

    ConsoleLogger logger;
    
    auto result = mlir_runner::run_mlir_postgres_table_scan("test_table", logger);
    
    EXPECT_TRUE(result) << "MLIR PostgreSQL table scan should succeed";
    

    g_mock_scan_context = nullptr;
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 