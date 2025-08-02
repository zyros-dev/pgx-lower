// Test file for terminator validation framework
// This file tests the LingoDB-style terminator utility infrastructure

#include "dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowUtilities.h"
#include "core/logging.h"

#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace subop_to_control_flow;

class TerminatorValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        context->loadDialect<mlir::scf::SCFDialect>();
        context->loadDialect<mlir::func::FuncDialect>();
        
        builder = std::make_unique<mlir::OpBuilder>(context.get());
        loc = builder->getUnknownLoc();
    }

    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<mlir::OpBuilder> builder;
    mlir::Location loc;
};

TEST_F(TerminatorValidationTest, TestTerminatorUtilsBasicValidation) {
    PGX_INFO("Testing basic terminator validation");
    
    // Create a simple function for testing
    auto funcType = mlir::FunctionType::get(context.get(), {}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "test_func", funcType);
    
    // Add a block without terminator
    auto* block = funcOp.addEntryBlock();
    
    // Test validation
    EXPECT_FALSE(TerminatorUtils::hasTerminator(*block));
    
    // Add terminator using our utilities
    builder->setInsertionPointToEnd(block);
    TerminatorUtils::createContextAppropriateTerminator(block, *builder, loc);
    
    // Verify terminator was added
    EXPECT_TRUE(TerminatorUtils::hasTerminator(*block));
    EXPECT_TRUE(TerminatorUtils::isValidTerminator(block->getTerminator()));
}

TEST_F(TerminatorValidationTest, TestRuntimeCallTermination) {
    PGX_INFO("Testing runtime call termination utilities");
    
    // Create a function with a PostgreSQL runtime call
    auto funcType = mlir::FunctionType::get(context.get(), {}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "test_pg_func", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create a mock PostgreSQL call
    auto calleeType = mlir::FunctionType::get(context.get(), {}, {});
    auto callOp = builder->create<mlir::func::CallOp>(loc, "StoreIntResult", calleeType.getResults(), mlir::ValueRange{});
    
    // Test PostgreSQL call detection
    EXPECT_TRUE(RuntimeCallTermination::isPostgreSQLRuntimeCall(callOp));
    
    // Apply termination
    RuntimeCallTermination::ensurePostgreSQLCallTermination(callOp, *builder, loc);
    
    // Verify termination was applied
    EXPECT_TRUE(block->getTerminator() != nullptr);
}

TEST_F(TerminatorValidationTest, TestBlockTerminationValidator) {
    PGX_INFO("Testing block termination validator");
    
    // Create a function for testing
    auto funcType = mlir::FunctionType::get(context.get(), {}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "test_validator", funcType);
    auto* block = funcOp.addEntryBlock();
    
    // Test validation of block without terminator
    EXPECT_FALSE(DefensiveProgramming::BlockTerminationValidator::validateBlock(block));
    
    // Repair the block
    mlir::PatternRewriter patternRewriter(*builder);
    DefensiveProgramming::BlockTerminationValidator::repairBlock(block, patternRewriter, loc);
    
    // Verify block is now valid
    EXPECT_TRUE(DefensiveProgramming::BlockTerminationValidator::validateBlock(block));
}

TEST_F(TerminatorValidationTest, TestAdvancedTerminatorProcessing) {
    PGX_INFO("Testing advanced terminator processing");
    
    // Create a function for testing
    auto funcType = mlir::FunctionType::get(context.get(), {}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "test_advanced", funcType);
    
    // Test operation termination
    mlir::PatternRewriter patternRewriter(*builder);
    AdvancedTerminatorProcessing::ensureOperationTermination(funcOp, patternRewriter, loc);
    
    // Verify function has proper termination
    EXPECT_TRUE(AdvancedTerminatorProcessing::hasProperTermination(&funcOp.getBody().front()));
}

TEST_F(TerminatorValidationTest, TestPostgreSQLIntegration) {
    PGX_INFO("Testing PostgreSQL integration utilities");
    
    // Create a function for testing
    auto funcType = mlir::FunctionType::get(context.get(), {}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "test_postgresql", funcType);
    auto* block = funcOp.addEntryBlock();
    
    // Test PostgreSQL compatible termination
    mlir::PatternRewriter patternRewriter(*builder);
    PostgreSQLIntegration::ensurePostgreSQLCompatibleTermination(block, patternRewriter, loc);
    
    // Verify termination was applied
    EXPECT_TRUE(block->getTerminator() != nullptr);
}

// Integration test for the complete terminator framework
TEST_F(TerminatorValidationTest, TestCompleteFrameworkIntegration) {
    PGX_INFO("Testing complete terminator framework integration");
    
    // Create a complex operation hierarchy
    auto funcType = mlir::FunctionType::get(context.get(), {}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "test_integration", funcType);
    auto* block = funcOp.addEntryBlock();
    
    builder->setInsertionPointToEnd(block);
    
    // Create nested control flow
    auto condition = builder->create<mlir::arith::ConstantIntOp>(loc, 1, builder->getI1Type());
    auto ifOp = builder->create<mlir::scf::IfOp>(loc, condition, false);
    
    // Test systematic termination
    mlir::PatternRewriter patternRewriter(*builder);
    
    // Apply comprehensive termination
    AdvancedTerminatorProcessing::ensureOperationTermination(funcOp, patternRewriter, loc);
    
    // Apply runtime call safety
    RuntimeCallTermination::applyRuntimeCallSafetyToOperation(funcOp, patternRewriter);
    
    // Apply PostgreSQL integration
    PostgreSQLIntegration::handleMemoryContextInvalidation(block, patternRewriter);
    
    // Validate all blocks
    funcOp.walk([](mlir::Operation* op) {
        for (auto& region : op->getRegions()) {
            for (auto& block : region.getBlocks()) {
                EXPECT_TRUE(DefensiveProgramming::BlockTerminationValidator::validateBlock(&block));
            }
        }
    });
    
    PGX_INFO("Complete framework integration test passed");
}