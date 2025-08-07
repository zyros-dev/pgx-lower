#include <gtest/gtest.h>
#include <llvm/Config/llvm-config.h>
#include <mlir/IR/MLIRContext.h>
#include <execution/mlir_runner.h>
#include <execution/mlir_logger.h>
#include <execution/logging.h>
#include "../test_helpers.h"
#include <signal.h>

// PostgreSQL headers for testing PlannedStmt and EState integration
extern "C" {
    // Forward declarations
    struct Plan;
    struct List;
    
    struct PlannedStmt {
        int dummy; // Mock structure for testing
        struct Plan* planTree; // This field is accessed by AST translator
        struct List* rtable; // Range table list - also accessed
    };
    
    struct Plan {
        int type; // Plan node type - T_SeqScan, etc.
    };
    
    struct EState {
        int dummy; // Mock structure for testing
    };
    
    struct ExprContext {
        int dummy; // Mock structure for testing
    };
    
    struct List {
        int dummy; // Mock list structure
    };
}

// Current dialect headers (using pgx-lower includes)
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <mlir/Dialect/DB/IR/DBDialect.h>
#include <mlir/Dialect/DSA/IR/DSADialect.h>

class MLIRRunnerTest : public ::testing::Test {
   protected:
    ConsoleLogger logger;
    
    void SetUp() override {
        // Setup for each test
        // Initialize logging for consistent test output
    }

    void TearDown() override {
        // Cleanup after each test
    }
};

// Test that MLIR context can be created and dialects loaded without TypeID errors
TEST_F(MLIRRunnerTest, DialectLoadingTest) {
    EXPECT_GT(LLVM_VERSION_MAJOR, 0);
    
    // Test that we can create an MLIR context and load our dialects
    // This should not produce undefined symbol errors for TypeIDs
    mlir::MLIRContext context;
    
    // Load each dialect individually to verify TypeID registration
    EXPECT_NO_THROW({
        auto* relalg_dialect = context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
        EXPECT_NE(relalg_dialect, nullptr);
    });
    
    EXPECT_NO_THROW({
        auto* db_dialect = context.getOrLoadDialect<pgx::db::DBDialect>();
        EXPECT_NE(db_dialect, nullptr);
    });
    
    EXPECT_NO_THROW({
        auto* dsa_dialect = context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
        EXPECT_NE(dsa_dialect, nullptr);
    });
}

// Test the basic MLIR runner function with null input
TEST_F(MLIRRunnerTest, NullPlannedStmtHandling) {
    // Test that the runner handles null PlannedStmt gracefully
    bool result = mlir_runner::run_mlir_postgres_ast_translation(nullptr, logger);
    EXPECT_FALSE(result); // Should return false for null input
}

// Test the MLIR runner with mock PlannedStmt
TEST_F(MLIRRunnerTest, MockPlannedStmtProcessing) {
    // Create a mock PlannedStmt for testing
    PlannedStmt mock_stmt;
    mock_stmt.dummy = 123;
    mock_stmt.planTree = nullptr; // Mock data - will cause AST translation to fail gracefully
    mock_stmt.rtable = nullptr; // Mock data - no range table
    
    // Test that the runner processes the mock statement
    // With the new full pipeline implementation, mock data will fail at AST translation
    // This is expected behavior - the runner now attempts full compilation
    bool result = mlir_runner::run_mlir_postgres_ast_translation(&mock_stmt, logger);
    EXPECT_FALSE(result); // Should return false because mock PlannedStmt cannot be translated
}

// Test the EState variant with null inputs
TEST_F(MLIRRunnerTest, NullEStateHandling) {
    // Test null PlannedStmt
    EState mock_estate;
    bool result = mlir_runner::run_mlir_with_estate(nullptr, &mock_estate, nullptr, logger);
    EXPECT_FALSE(result);
    
    // Test null EState
    PlannedStmt mock_stmt;
    result = mlir_runner::run_mlir_with_estate(&mock_stmt, nullptr, nullptr, logger);
    EXPECT_FALSE(result);
}

// Test the EState variant with mock inputs
TEST_F(MLIRRunnerTest, MockEStateProcessing) {
    // Create mock structures for testing
    PlannedStmt mock_stmt;
    mock_stmt.dummy = 123;
    mock_stmt.planTree = nullptr; // Mock data - will cause AST translation to fail gracefully
    mock_stmt.rtable = nullptr; // Mock data - no range table
    
    EState mock_estate;
    mock_estate.dummy = 456;
    
    ExprContext mock_econtext;
    mock_econtext.dummy = 789;
    
    // Test that the runner processes the mock structures
    // With the new full pipeline implementation, mock data will fail at AST translation
    // This is expected behavior - the runner now attempts full compilation
    bool result = mlir_runner::run_mlir_with_estate(&mock_stmt, &mock_estate, &mock_econtext, logger);
    EXPECT_FALSE(result); // Should return false because mock PlannedStmt cannot be translated
}

// Test that multiple calls don't interfere with each other
TEST_F(MLIRRunnerTest, MultipleCalls) {
    PlannedStmt mock_stmt;
    mock_stmt.dummy = 123;
    mock_stmt.planTree = nullptr; // Mock data - will cause AST translation to fail gracefully
    mock_stmt.rtable = nullptr; // Mock data - no range table
    
    // Multiple calls should all fail consistently (mock data cannot be translated)
    // This verifies that the runner doesn't maintain improper state between calls
    EXPECT_FALSE(mlir_runner::run_mlir_postgres_ast_translation(&mock_stmt, logger));
    EXPECT_FALSE(mlir_runner::run_mlir_postgres_ast_translation(&mock_stmt, logger));
    EXPECT_FALSE(mlir_runner::run_mlir_postgres_ast_translation(&mock_stmt, logger));
}

// Test that demonstrates the full pipeline behavior
TEST_F(MLIRRunnerTest, FullPipelineBehavior) {
    // This test documents the expected behavior of the new full pipeline implementation
    // When given invalid mock data, the pipeline should fail at AST translation step
    
    PlannedStmt mock_stmt;
    mock_stmt.dummy = 123;
    mock_stmt.planTree = nullptr; // Mock data - will cause AST translation to fail gracefully
    mock_stmt.rtable = nullptr; // Mock data - no range table
    
    // The runner now attempts full compilation pipeline:
    // 1. Initialize MLIR context and dialects ✓
    // 2. Translate PostgreSQL AST to RelAlg MLIR ✗ (fails with mock data)
    // 3. RelAlg → DB dialect lowering (not reached)
    // 4. DB → DSA dialect lowering (not reached) 
    // 5. Final verification (not reached)
    
    bool result = mlir_runner::run_mlir_postgres_ast_translation(&mock_stmt, logger);
    EXPECT_FALSE(result); // Expected to fail at AST translation with mock data
    
    // TODO Phase 5: Add tests with proper PlannedStmt structures that can be translated
    // These would test the full pipeline: AST → RelAlg → DB → DSA → (future: LLVM IR → JIT)
}