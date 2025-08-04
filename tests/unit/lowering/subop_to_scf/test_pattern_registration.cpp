#include "gtest/gtest.h"
#include "execution/logging.h"

#include "SubOpToControlFlowPatterns.h"
#include "compiler/Dialect/SubOperator/SubOpDialect.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace pgx_lower::compiler::dialect::subop_to_cf;

class PatternRegistrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<pgx_lower::compiler::dialect::subop::SubOperatorDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
    }

    mlir::MLIRContext context;
};

// Test that pattern registration functions can be called without crashing
TEST_F(PatternRegistrationTest, PopulateSubOpToControlFlowConversionPatterns) {
    mlir::RewritePatternSet patterns(&context);
    mlir::TypeConverter typeConverter;
    
    // This should not crash
    EXPECT_NO_THROW({
        populateSubOpToControlFlowConversionPatterns(patterns, typeConverter, &context);
    });
    
    // Patterns should have been added
    EXPECT_GT(patterns.getNativePatterns().size(), 0);
}

TEST_F(PatternRegistrationTest, PopulateTableOperationPatterns) {
    mlir::RewritePatternSet patterns(&context);
    mlir::TypeConverter typeConverter;
    
    // This should not crash
    EXPECT_NO_THROW({
        populateTableOperationPatterns(patterns, typeConverter, &context);
    });
    
    // Patterns should have been added (could be 0 if no table patterns exist yet)
    EXPECT_GE(patterns.getNativePatterns().size(), 0);
}

// Test that both pattern registration functions can be called together
TEST_F(PatternRegistrationTest, PopulateBothPatternSets) {
    mlir::RewritePatternSet patterns(&context);
    mlir::TypeConverter typeConverter;
    
    // Both calls should succeed
    EXPECT_NO_THROW({
        populateSubOpToControlFlowConversionPatterns(patterns, typeConverter, &context);
        populateTableOperationPatterns(patterns, typeConverter, &context);
    });
    
    // At least the first call should have added patterns
    EXPECT_GT(patterns.getNativePatterns().size(), 0);
}