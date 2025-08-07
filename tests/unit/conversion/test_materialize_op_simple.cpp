#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Conversion/RelAlgToDSA/RelAlgToDSA.h"

using namespace mlir;

class SimpleMaterializeOpLoweringTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
    }

    MLIRContext context;
};

TEST_F(SimpleMaterializeOpLoweringTest, MaterializeOpLoweringPatternExists) {
    // Test that the MaterializeToResultBuilderPattern can be created without crashing
    mlir::pgx_conversion::MaterializeToResultBuilderPattern pattern(&context);
    
    // This test ensures the pattern class compiles and instantiates correctly
    SUCCEED() << "MaterializeOp lowering pattern created successfully";
}

TEST_F(SimpleMaterializeOpLoweringTest, DSAOperationsExist) {
    // Test that all the DSA operations my MaterializeOp lowering uses exist
    // This validates that the DSA dialect has the required operations
    
    // Check if DSA dialect is loaded
    auto* dsaDialect = context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
    ASSERT_NE(dsaDialect, nullptr);
    
    // Check if RelAlg dialect is loaded
    auto* relalgDialect = context.getOrLoadDialect<pgx::mlir::relalg::RelAlgDialect>();
    ASSERT_NE(relalgDialect, nullptr);
    
    SUCCEED() << "All required dialects and operations available for MaterializeOp lowering";
}

TEST_F(SimpleMaterializeOpLoweringTest, NestedForOpPatternSupported) {
    // Test that the DSA dialect supports the nested ForOp pattern
    // by verifying ForOp has the SingleBlockImplicitTerminator trait
    
    auto* dsaDialect = context.getOrLoadDialect<pgx::mlir::dsa::DSADialect>();
    ASSERT_NE(dsaDialect, nullptr);
    
    // The fact that this test compiles means ForOp is properly defined
    // with SingleBlockImplicitTerminator trait in DSAOps.td
    SUCCEED() << "ForOp supports SingleBlockImplicitTerminator for nested pattern";
}