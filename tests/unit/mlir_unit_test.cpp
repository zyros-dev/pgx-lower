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
#include <core/query_analyzer.h>
#include <core/error_handling.h>
#include <core/mlir_code_generator.h>
#include <dialects/pg/PgDialect.h>
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

extern "C" int32_t get_int_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    // Mock implementation for unit tests
    *is_null = false;
    return field_index * 42; // Return predictable values
}

extern "C" int64_t get_text_field(void* tuple_handle, int32_t field_index, bool* is_null) {
    // Mock implementation for unit tests
    static const char* mock_text = "mock_text_field";
    *is_null = false;
    return reinterpret_cast<int64_t>(mock_text);
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

TEST(MLIRTest, QueryAnalyzer) {
    using namespace pgx_lower;
    
    // Test simple SELECT query
    auto caps1 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test");
    EXPECT_TRUE(caps1.requiresSeqScan);
    EXPECT_FALSE(caps1.requiresFilter);
    EXPECT_TRUE(caps1.isMLIRCompatible());
    
    // Test SELECT with WHERE
    auto caps2 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test WHERE id > 5");
    EXPECT_TRUE(caps2.requiresSeqScan);
    EXPECT_TRUE(caps2.requiresFilter);
    EXPECT_FALSE(caps2.isMLIRCompatible());
    
    // Test SELECT with JOIN
    auto caps3 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test JOIN other ON test.id = other.id");
    EXPECT_TRUE(caps3.requiresSeqScan);
    EXPECT_TRUE(caps3.requiresJoin);
    EXPECT_FALSE(caps3.isMLIRCompatible());
    
    // Test SELECT with ORDER BY
    auto caps4 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test ORDER BY id");
    EXPECT_TRUE(caps4.requiresSeqScan);
    EXPECT_TRUE(caps4.requiresSort);
    EXPECT_FALSE(caps4.isMLIRCompatible());
    
    // Test SELECT with LIMIT
    auto caps5 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test LIMIT 10");
    EXPECT_TRUE(caps5.requiresSeqScan);
    EXPECT_TRUE(caps5.requiresLimit);
    EXPECT_FALSE(caps5.isMLIRCompatible());
    
    // Test SELECT with aggregation
    auto caps6 = QueryAnalyzer::analyzeForTesting("SELECT COUNT(*) FROM test");
    EXPECT_TRUE(caps6.requiresSeqScan);
    EXPECT_TRUE(caps6.requiresAggregation);
    EXPECT_FALSE(caps6.isMLIRCompatible());
}

// TODO: Fix segfault in PostgreSQL dialect generation test
// TEST(MLIRTest, PostgreSQLDialectGeneration) {
//     using namespace pgx_lower;
//     
//     // Test the new PostgreSQL dialect infrastructure
//     mlir::MLIRContext context;
//     
//     // Register the pg dialect
//     context.getOrLoadDialect<mlir::pg::PgDialect>();
//     
//     // Create a modular MLIR generator
//     ModularMLIRGenerator generator(&context);
//     
//     // Generate MLIR with pg dialect (this should show the before/after transformation)
//     auto func = generator.generateTableScanFunction("test_table");
//     
//     // Verify the function was created
//     EXPECT_TRUE(func);
//     EXPECT_EQ(func.getName(), "main");
//     
//     // The debug output should show the transformation in the console
//     std::cout << "[TEST] PostgreSQL dialect transformation completed!" << std::endl;
// }

TEST(MLIRTest, ErrorHandling) {
    using namespace pgx_lower;
    
    // Test error creation and formatting
    auto error = ErrorManager::queryAnalysisError("Test error message", "SELECT * FROM test");
    EXPECT_EQ(error.severity, ErrorSeverity::ERROR_LEVEL);
    EXPECT_EQ(error.category, ErrorCategory::QUERY_ANALYSIS);
    EXPECT_EQ(error.message, "Test error message");
    EXPECT_TRUE(error.context.find("SELECT * FROM test") != std::string::npos);
    
    // Test formatted message
    std::string formatted = error.getFormattedMessage();
    EXPECT_TRUE(formatted.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(formatted.find("[QUERY_ANALYSIS]") != std::string::npos);
    EXPECT_TRUE(formatted.find("Test error message") != std::string::npos);
    
    // Test Result type for success
    Result<int> successResult(42);
    EXPECT_TRUE(successResult.isSuccess());
    EXPECT_FALSE(successResult.isError());
    EXPECT_EQ(successResult.getValue(), 42);
    
    // Test Result type for error
    auto errorInfo = ErrorManager::makeError(ErrorSeverity::ERROR_LEVEL, ErrorCategory::EXECUTION, "Test failure");
    Result<int> errorResult(errorInfo);
    EXPECT_FALSE(errorResult.isSuccess());
    EXPECT_TRUE(errorResult.isError());
    EXPECT_EQ(errorResult.valueOr(99), 99);
    
    // Test console error handler
    ErrorManager::setHandler(std::make_unique<ConsoleErrorHandler>());
    auto handler = ErrorManager::getHandler();
    EXPECT_NE(handler, nullptr);
    EXPECT_FALSE(handler->shouldAbortOnError(error));
    
    // Test fatal error should abort
    auto fatalError = ErrorManager::makeError(ErrorSeverity::FATAL_LEVEL, ErrorCategory::EXECUTION, "Fatal error");
    EXPECT_TRUE(handler->shouldAbortOnError(fatalError));
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 